# gen_debiased_nli/mnli_z_aug

Classification · `pietrolesci/gen_debiased_nli`

<details><summary>Example 1</summary>

published Jev · train · gen-debiased-nli-mnli-z-aug-b68f91c293:train:2763

**State**

<pre>text_A: Choose your own fish from the display and find out how much it will cost before having it cooked.
text_B: Pick which fish you want and learn the price before having it cooked.</pre>

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

published Jev · train · gen-debiased-nli-mnli-z-aug-b68f91c293:train:2763:noul-label-verification

**State**

<pre>text_A: Choose your own fish from the display and find out how much it will cost before having it cooked.
text_B: Pick which fish you want and learn the price before having it cooked.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither? Is &quot;neutral&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · gen-debiased-nli-mnli-z-aug-b68f91c293:train:4063

**State**

<pre>text_A: We need to discover what unites these eight states and how they can get beyond their differences to reach a common ground.
text_B: There are certain states where there is almost no diversity.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>
