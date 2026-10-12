# conj_nli

Classification · `pietrolesci/conj_nli`

<details><summary>Example 1</summary>

published Jev · train · conj-nli-0f0ab95726:train:40

**State**

<pre>text_A: The series portrays her as a scheming and secretive but caring woman who develops into a formidable antagonist as her agenda against Serena van der Woodsen progressed.
text_B: The series portrays her as a scheming and secretive woman who develops into a formidable antagonist as her agenda against Serena van der Woodsen progressed.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · conj-nli-0f0ab95726:train:40:choice-criteria-permutation

**State**

<pre>text_A: The series portrays her as a scheming and secretive but caring woman who develops into a formidable antagonist as her agenda against Serena van der Woodsen progressed.
text_B: The series portrays her as a scheming and secretive woman who develops into a formidable antagonist as her agenda against Serena van der Woodsen progressed.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. contradiction
2. entailment
3. neutral</pre>

**Gold target (weight)**

<pre>2. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · conj-nli-0f0ab95726:train:13690

**State**

<pre>text_A: The Commission does have the ability to subpoena documents and witnesses for testimony, a power that the Pecora Commission had but the 9/11 Commission did not.
text_B: The Commission does have the ability to subpoena documents and witnesses for testimony, a power that the Pecora Commission had.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
