# babi_nli/simple-negation

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-simple-negation-96c05c19e2:train:2

**State**

<pre>text_A: Mary is no longer in the bedroom. Daniel moved to the hallway. Sandra moved to the bedroom. Sandra is in the bathroom. Daniel is not in the hallway. Daniel is in the office.
text_B: Sandra is in the office.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-simple-negation-96c05c19e2:train:2:noul-label-verification

**State**

<pre>text_A: Mary is no longer in the bedroom. Daniel moved to the hallway. Sandra moved to the bedroom. Sandra is in the bathroom. Daniel is not in the hallway. Daniel is in the office.
text_B: Sandra is in the office.</pre>

**Question**

<pre>Does text_A entail text_B? Is &quot;not-entailed&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-simple-negation-96c05c19e2:train:623

**State**

<pre>text_A: Mary is not in the office. Daniel went to the kitchen. John travelled to the garden. Sandra is in the office. John is in the bedroom. John is no longer in the bedroom. Mary is no longer in the kitchen. Sandra went to the hallway.
text_B: John is in the bedroom.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
