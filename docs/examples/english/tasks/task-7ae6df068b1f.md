# breaking_nli

Classification · `pietrolesci/breaking_nli`

<details><summary>Example 1</summary>

published Jev · train · breaking-nli-9bb3f705a7:train:pack-941714a0721a:label-B

**State**

<pre>Item A:
text_A: kids are cleaning the bathroom.
text_B: kids are cleaning the courtroom.

Item B:
text_A: A boy runs near some steps.
text_B: A daughter runs near some steps.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · breaking-nli-9bb3f705a7:train:pack-941714a0721a:same-A-B

**State**

<pre>Item A:
text_A: kids are cleaning the bathroom.
text_B: kids are cleaning the courtroom.

Item B:
text_A: A boy runs near some steps.
text_B: A daughter runs near some steps.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Do Item A and Item B have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · breaking-nli-9bb3f705a7:train:pack-941714a0721a:exists-1

**State**

<pre>Item A:
text_A: kids are cleaning the bathroom.
text_B: kids are cleaning the courtroom.

Item B:
text_A: A boy runs near some steps.
text_B: A daughter runs near some steps.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Does at least one item have the label &quot;neutral&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
