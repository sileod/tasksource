"""The chat model behind tasksource's LLM stages: audits, the discovery web agent, synthetic generation.

Configured like any OpenAI client: ``OPENAI_BASE_URL``, ``OPENAI_API_KEY`` and ``OPENAI_MODEL`` (or
``--model``). OpenAI, OpenRouter, vLLM and other compatible servers work directly. A backend that needs
an adapter runs behind a local bridge, e.g. harness-link: ``albert exec -- python -m ...``.

Typed-decision (Jev) calls use OpenRouter's decisions API and are configured separately.
"""
import os

KEY_ENV = 'OPENAI_API_KEY'


def base_url():
    return os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1').rstrip('/')


def model(explicit=None):
    name = explicit or os.environ.get('OPENAI_MODEL')
    if not name:
        raise SystemExit('Set OPENAI_MODEL or pass --model (any model of the OpenAI-compatible endpoint).')
    return name


def litlm_model(explicit=None):
    """The model as a litlm route: litellm's OpenAI handler reads OPENAI_BASE_URL itself.
    Explicit litlm routes (``openrouter/...``, ``direct/...``) pass through unchanged."""
    name = model(explicit)
    return name if name.startswith(LITLM_ROUTES) else 'direct/openai/' + name


LITLM_ROUTES = ('direct/', 'openrouter/', 'nvidia_nim/', 'albert/')
