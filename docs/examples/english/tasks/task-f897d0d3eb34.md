# babi_nli/basic-coreference

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-basic-coreference-ece85559d7:train:24

**State**

<pre>text_A: Mary journeyed to the office. After that she travelled to the kitchen. Daniel moved to the kitchen. Afterwards he journeyed to the hallway.
text_B: Mary is in the hallway.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-basic-coreference-ece85559d7:train:24:noul-label-verification

**State**

<pre>text_A: Mary journeyed to the office. After that she travelled to the kitchen. Daniel moved to the kitchen. Afterwards he journeyed to the hallway.
text_B: Mary is in the hallway.</pre>

**Question**

<pre>Does text_A entail text_B? Is &quot;entailed&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-basic-coreference-ece85559d7:train:593

**State**

<pre>text_A: John went to the bathroom. Afterwards he travelled to the office.
text_B: John is in the office.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
