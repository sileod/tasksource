"""Optional LLM critic: cross-question coherence checks.

Deterministic validation cannot tell whether all questions genuinely
refer to the same state or test distinct skills; the critic can.
The critic uses its OWN provider config (see CriticConfig), so e.g.
Luna generation + Albert/DeepSeek critic works correctly.

Critic calls are cached like generation calls:
    hash(critic model + temperature + prompt hash + canonical bundle)
and raw API responses are preserved.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from pathlib import Path

from . import providers
from .generate import PROMPTS_DIR, extract_json_object, prompt_hash
from .specs import SKILL_DEFINITIONS

_SKILL_CUES = {
    "policy_violation": re.compile(r"\b(?:violat|comply|compliance|breach|follow|permitted|allowed|against (?:the )?(?:policy|rule))\w*\b", re.I),
    "sla_breach": re.compile(r"\b(?:sla|deadline|response (?:time|window)|breach|late|on time|timely|within \d)\b", re.I),
    "fraud_likelihood": re.compile(r"\b(?:fraud|fraudulent|decept|misrepresent|improper|legitimate)\b", re.I),
    "churn_risk": re.compile(r"\b(?:churn|cancel|leav|switch|renew|non-renew|retention|relationship|remain a customer|lose (?:the )?customer)\w*\b", re.I),
    "groundedness": re.compile(r"\b(?:grounded|supported|support|evidence|follows from|according to|suggest|indicat)\w*\b", re.I),
    "completeness": re.compile(r"\b(?:complete|comprehensive|all required|all necessary|missing (?:step|information)|sufficient)\b", re.I),
    "access_justification": re.compile(r"\b(?:access|authoriz|approval|justif|permission)\b", re.I),
    "topic_classification": re.compile(r"\b(?:topic|subject|mainly about|primarily about)\b", re.I),
    "document_purpose": re.compile(r"\b(?:purpose|why (?:was|is).*(?:written|sent)|function of (?:this|the) (?:document|email|message|report))\b", re.I),
    "main_point": re.compile(r"\b(?:main point|central (?:idea|message)|key takeaway|primary claim)\b", re.I),
    "audience_inference": re.compile(r"\b(?:audience|addressed to|intended for|recipient)\b", re.I),
    "argument_role": re.compile(r"\b(?:role|function|claim|support|caveat|request)\b", re.I),
    "message_tone": re.compile(r"\b(?:tone|comes across|wording sound)\b", re.I),
}

_SPECIALIZED_DOMAIN_CUES = {
    "education": re.compile(r"\b(?:school|student|teacher|course|classroom|college|university|academic|education|campus|curricul)\w*\b", re.I),
    "research": re.compile(r"\b(?:research|study|experiment|laborator|\blab\b|dataset|academic|scientif|university|hypothesis|participant)\w*\b", re.I),
    "pharma": re.compile(r"\b(?:pharma|drug|medication|dose|clinical trial|adverse event|patient|regulatory)\w*\b", re.I),
    "healthcare_ops": re.compile(r"\b(?:patient|clinic|hospital|medical|healthcare|nurse|physician|appointment|care team)\w*\b", re.I),
    "hr_recruiting": re.compile(r"\b(?:candidate|recruit|hiring|interview|job offer|applicant|human resources|\bHR\b)\w*\b", re.I),
    "legal_compliance": re.compile(r"\b(?:legal|compliance|regulat|counsel|contract|policy|audit|law|statute)\w*\b", re.I),
}


def deterministic_critic_issues(bundle: dict) -> list[str]:
    """Catch objective semantic mismatches that an agreeable LLM may overlook."""
    issues: list[str] = []
    state = bundle.get("state", "")
    state_lower = state.casefold()
    policy_in_state = bool(re.search(
        r"\b(?:policy|procedure|protocol|rule|requires?|must|shall|SOP[- ]?\d*)\b",
        state, re.I))
    domain = bundle.get("domain", "")
    domain_cue = _SPECIALIZED_DOMAIN_CUES.get(domain)
    if domain_cue is not None and not domain_cue.search(state):
        issues.append(f"state has no evidence of declared domain {domain}")
    for question in bundle.get("questions", []):
        qid = question.get("question_id", "unknown")
        skill = question.get("skill", "")
        text = question.get("question", "")
        cue = _SKILL_CUES.get(skill)
        if cue is not None and not cue.search(text):
            issues.append(f"{qid}: question does not test declared skill {skill}")
        if skill == "toxicity" and not re.search(
                r"\b(?:toxic(?:ity)?(?: level)? (?:of|in) (?:the )?(?:language|wording|message|email|comment)|"
                r"language.*toxic|toxic.*language|abusive|hostile|hate speech|insult|personal attack)\b",
                text, re.I):
            issues.append(f"{qid}: toxicity must assess harmful language, not subject-matter toxicity")
        if skill == "data_sensitivity":
            confidentiality = re.search(
                r"\b(?:confidential(?:ity)?|restricted|private|public|"
                r"sensitive (?:data|information|record)|classified as sensitive|"
                r"data sensitivity|disclos|expos|PII|personal data|classification)\b", text, re.I)
            if not confidentiality or "data quality" in text.casefold():
                issues.append(f"{qid}: data_sensitivity must assess confidentiality or disclosure")
        if skill == "sentiment" and re.search(
                r"\b(?:monitoring|dashboard|system|automated) (?:alert|notification)\b",
                state, re.I) and not re.search(
                    r"\b(?:person|customer|user|agent|speaker|writer|author|message|comment|"
                    r"email|review|says?|said|express)\w*\b", text, re.I):
            issues.append(f"{qid}: sentiment must assess a person's expressed evaluation")
        if skill == "policy_violation" and not policy_in_state:
            issues.append(f"{qid}: policy_violation lacks a policy or rule in the state")
        if re.search(r"\b(?:proper procedure|under (?:the )?policy|policy violation)\b", text, re.I) \
                and not policy_in_state:
            issues.append(f"{qid}: question requires an unstated policy or procedure")
        if question.get("format") == "score":
            options = list(question.get("options", []))
            numeric = options and all(re.fullmatch(r"-?\d+", str(option)) for option in options)
            if len(options) > 5 and numeric:
                issues.append(f"{qid}: numeric score has more than five weakly anchored levels")
            if numeric:
                lo, hi = re.escape(str(options[0])), re.escape(str(options[-1]))
                marker = r"(?:\s+(?:means|for)\b|\s*(?:=|:|\())"
                has_lo = re.search(rf"\b{lo}\b{marker}", text, re.I)
                has_hi = re.search(rf"\b{hi}\b{marker}", text, re.I)
                if not has_lo or not has_hi:
                    issues.append(f"{qid}: numeric score does not define both endpoint meanings")
    return issues


def load_critic_prompt(version: str) -> str:
    return (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8").replace(
        "{{SKILL_DEFINITIONS}}", json.dumps(SKILL_DEFINITIONS, indent=2))


def critic_cache_key(model: str, temperature: float, prompt: str, bundle: dict, endpoint: str = "") -> str:
    """``endpoint`` names the provider (``name@base_url``): one model name can be served by several."""
    canonical = json.dumps(
        {"state_id": bundle.get("state_id"), "domain": bundle.get("domain"),
         "state": bundle.get("state"),
         "questions": bundle.get("questions")},
        sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(
        f"{endpoint}\n{model}\n{temperature}\n{prompt_hash(prompt)}\n{canonical}".encode("utf-8")
    ).hexdigest()


def mock_critique(bundle: dict) -> dict:
    texts = [q.get("question", "") for q in bundle.get("questions", [])]
    issues = []
    if len(set(texts)) != len(texts):
        issues.append("questions are paraphrases of each other")
    return {"pass": not issues, "issues": issues, "score": 1.0 if not issues else 0.0}


def checked_verdict(response: dict, bundle: dict) -> dict:
    """Gate on structured checks rather than contradictory critic prose."""
    model_issues = [str(issue) for issue in response.get("issues", [])]
    issues: list[str] = []
    checks = response.get("checks")
    questions = bundle.get("questions", [])
    expected = {q["question_id"] for q in questions}
    bundle_checks = response.get("bundle_checks")
    for field in ("same_state", "distinct_questions", "consistent", "domain_match"):
        if not isinstance(bundle_checks, dict) or bundle_checks.get(field) is not True:
            issues.append(f"critic bundle {field} check failed")
    if not isinstance(checks, list) or len(checks) != len(questions) or {
            check.get("question_id") for check in checks if isinstance(check, dict)
    } != expected:
        issues.append("missing or duplicate per-question critic checks")
    else:
        state = re.sub(r"\s+", " ", bundle.get("state", "")).casefold()
        for check in checks:
            qid = check["question_id"]
            question = next(q for q in questions if q["question_id"] == qid)
            quote = check.get("evidence_quote")
            if not isinstance(quote, str) or not quote.strip() or re.sub(
                    r"\s+", " ", quote).strip().casefold() not in state:
                issues.append(f"{qid}: evidence quote not found in state")
            elif question.get("skill") == "sentiment" and not re.search(
                    r"\b(?:positive|negative|happy|pleased|satisf|glad|grateful|hopeful|"
                    r"negative|frustrat|angry|upset|disappoint|unhappy|ridiculous|"
                    r"unacceptable|hate|love|worr|concern|regret|sorry|terrible|great)\w*\b",
                    quote, re.I):
                issues.append(f"{qid}: evidence quote does not express sentiment")
            for field in ("skill_match", "supported", "unique_answer"):
                if check.get(field) is not True:
                    issues.append(f"{qid}: critic {field} check failed")
            if check.get("inferred_skill") != question.get("skill"):
                issues.append(
                    f"{qid}: inferred skill {check.get('inferred_skill')!r} does not match "
                    f"declared skill {question.get('skill')!r}")
            if check.get("policy_sufficient") is not True:
                issues.append(f"{qid}: required policy or decision standard is missing")
            answer = check.get("answer")
            if question.get("format") == "choice" and answer not in question.get("options", []):
                issues.append(f"{qid}: critic did not select an exact choice option")
            elif question.get("format") == "choice" and question.get("skill") == "refund_approval" \
                    and isinstance(answer, str) and answer.casefold().startswith("yes") \
                    and not re.search(r"\b(?:refund|reimburse|repay|return (?:the )?(?:charge|payment|money))\w*\b",
                                      answer, re.I):
                issues.append(f"{qid}: selected answer substitutes another remedy for a refund")
            elif question.get("format") == "noul" and not isinstance(answer, bool):
                issues.append(f"{qid}: critic did not return a boolean answer")
            elif question.get("format") == "score" and not (
                    isinstance(answer, int) and not isinstance(answer, bool)
                    and 0 <= answer < len(question.get("options", []))):
                issues.append(f"{qid}: critic did not return a valid score index")
    issues.extend(deterministic_critic_issues(bundle))
    issues = list(dict.fromkeys(issues))
    return {"pass": not issues, "issues": issues,
            "model_issues": model_issues, "checks": checks,
            "bundle_checks": bundle_checks,
            "score": float(response.get("score", 0.0))}


class RequestPacer:
    """Space critic calls so a batch stays below the provider's minute limit."""

    def __init__(self, requests_per_minute: int):
        self.interval = 60.0 / max(1, requests_per_minute)
        self.next_start = 0.0
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if now < self.next_start:
                await asyncio.sleep(self.next_start - now)
            self.next_start = time.monotonic() + self.interval


async def _critique_one(sem, client, model: str, temperature: float,
                        template: str, bundle: dict, raw_dir: Path,
                        pacer: RequestPacer | None = None, endpoint: str = "") -> dict:
    guard_issues = deterministic_critic_issues(bundle)
    if guard_issues:
        return {"state_id": bundle["state_id"], "pass": False,
                "issues": guard_issues, "model_issues": [], "score": 0.0}
    key = critic_cache_key(model, temperature, template, bundle, endpoint)
    cached = raw_dir / f"{key}.json"
    if cached.exists():
        record = json.loads(cached.read_text(encoding="utf-8"))
        verdict = dict(record["verdict"])
        if verdict.get("checks") is not None:
            response = dict(verdict)
            response["issues"] = list(verdict.get("model_issues", []))
            verdict = checked_verdict(response, bundle)
        else:
            issues = list(verdict.get("issues", [])) + deterministic_critic_issues(bundle)
            verdict["issues"] = list(dict.fromkeys(issues))
            verdict["pass"] = verdict.get("pass") is True and not verdict["issues"]
        return {"state_id": bundle["state_id"], **verdict}
    if client is None:
        verdict = mock_critique(bundle)
        guard_issues = deterministic_critic_issues(bundle)
        verdict["issues"] = list(dict.fromkeys(verdict["issues"] + guard_issues))
        verdict["pass"] = verdict["pass"] and not verdict["issues"]
        cached.write_text(json.dumps(
            {"cache_key": key, "state_id": bundle["state_id"],
             "provider": "mock", "model": model, "verdict": verdict},
            ensure_ascii=False, indent=2), encoding="utf-8")
        return {"state_id": bundle["state_id"], **verdict}
    payload = {"state_id": bundle["state_id"], "domain": bundle.get("domain"),
               "state": bundle["state"],
               "questions": bundle["questions"]}
    prompt = template.replace("{{BUNDLE_JSON}}", json.dumps(payload, ensure_ascii=False, indent=2))
    async with sem:
        if pacer is not None:
            await pacer.wait()
        result = await providers.chat_complete(
            client, model, [{"role": "user", "content": prompt}],
            temperature=temperature, max_tokens=2000)
    try:
        verdict = checked_verdict(extract_json_object(result["text"]), bundle)
    except (ValueError, json.JSONDecodeError, TypeError, KeyError):
        verdict = {"pass": False, "issues": ["unparseable critic response"], "score": 0.0}
    cached.write_text(json.dumps(
        {"cache_key": key, "state_id": bundle["state_id"],
         "provider": "critic", "requested_model": model,
         "returned_model": result["returned_model"],
         "raw_response": result["raw"], "raw_text": result["text"],
         "verdict": verdict},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return {"state_id": bundle["state_id"], **verdict}


async def critique_bundles_async(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not cfg.critic.enabled:
        return [{"state_id": b["state_id"], "pass": True, "issues": [], "score": 1.0} for b in bundles]
    template = load_critic_prompt(cfg.critic.prompt_version)
    provider = cfg.critic_provider()
    clients = [None]
    if provider.name != "mock":
        clients = [providers.make_client(provider, key)
                   for key in providers.available_api_keys(provider)]
    try:
        sem = asyncio.Semaphore(max(1, cfg.generation.concurrency))
        pacers = [RequestPacer(cfg.critic.requests_per_minute) if client is not None
                  else None for client in clients]
        out = await asyncio.gather(*[_critique_one(sem, clients[i % len(clients)], cfg.critic.model,
                                                   cfg.critic.temperature, template, b, raw_dir,
                                                   pacers[i % len(clients)],
                                                   f"{provider.name}@{provider.base_url.rstrip('/')}")
                                     for i, b in enumerate(bundles)])
    finally:
        for client in clients:
            if client is not None:
                try:
                    await client.close()
                except Exception:
                    pass
    return list(out)


def critique_bundles(cfg, bundles: list[dict], raw_dir: Path) -> list[dict]:
    return asyncio.run(critique_bundles_async(cfg, bundles, raw_dir))
