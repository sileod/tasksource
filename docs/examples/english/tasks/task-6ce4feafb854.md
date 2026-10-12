# babi_nli/indefinite-knowledge

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-indefinite-knowledge-ca5ffe07c8:train:6

**State**

<pre>text_A: Mary travelled to the kitchen. Fred is in the school. Fred went back to the office. Julie is either in the park or the office.
text_B: Mary is in the cinema.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-indefinite-knowledge-ca5ffe07c8:train:6:noul-label-verification

**State**

<pre>text_A: Mary travelled to the kitchen. Fred is in the school. Fred went back to the office. Julie is either in the park or the office.
text_B: Mary is in the cinema.</pre>

**Question**

<pre>Does text_A entail text_B? Is &quot;entailed&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-indefinite-knowledge-ca5ffe07c8:train:846

**State**

<pre>text_A: Julie moved to the cinema. Mary is either in the park or the park. Julie is either in the school or the cinema. Julie went to the bedroom.
text_B: Julie is in the bedroom.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
