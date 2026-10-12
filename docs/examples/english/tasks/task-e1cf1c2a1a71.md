# pragmeval/gum

Classification · `sileod/pragmeval`

<details><summary>Example 1</summary>

published Jev · train · pragmeval-gum-4abdfde4d7:train:221

**State**

<pre>text_A: &quot;when he responded to the altar call at his church &quot;&quot;&quot;
text_B: &quot;to accept christ as my saviour . &quot;&quot;&quot;</pre>

**Question**

<pre>Which discourse relation links text_B to text_A?</pre>

**Criteria (choice)**

<pre>1. preparation
2. evaluation
3. circumstance
4. solutionhood
5. justify
6. result
7. evidence
8. purpose
9. concession
10. elaboration
11. background
12. condition
13. cause
14. restatement
15. motivation
16. antithesis
17. no_relation</pre>

**Gold target (weight)**

<pre>8. purpose — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · pragmeval-gum-4abdfde4d7:train:221:noul-label-verification

**State**

<pre>text_A: &quot;when he responded to the altar call at his church &quot;&quot;&quot;
text_B: &quot;to accept christ as my saviour . &quot;&quot;&quot;</pre>

**Question**

<pre>Which discourse relation links text_B to text_A? Is &quot;evaluation&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · pragmeval-gum-4abdfde4d7:train:1615

**State**

<pre>text_A: deploy .
text_B: tips</pre>

**Question**

<pre>Which discourse relation links text_B to text_A?</pre>

**Criteria (choice)**

<pre>1. preparation
2. evaluation
3. circumstance
4. solutionhood
5. justify
6. result
7. evidence
8. purpose
9. concession
10. elaboration
11. background
12. condition
13. cause
14. restatement
15. motivation
16. antithesis
17. no_relation</pre>

**Gold target (weight)**

<pre>17. no_relation — 1</pre>

</details>
