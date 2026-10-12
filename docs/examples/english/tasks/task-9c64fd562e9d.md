# counterfactually-augmented-snli

Classification · `tasksource/counterfactually-augmented-snli`

<details><summary>Example 1</summary>

published Jev · train · counterfactually-augmented-snli-283b54ff0e:train:6

**State**

<pre>text_A: In a martial arts demonstration a man with two swords faces a woman with a Chinese spear while another woman watches.
text_B: The woman is watching.</pre>

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

published Jev · train · counterfactually-augmented-snli-283b54ff0e:train:6:noul-label-verification

**State**

<pre>text_A: In a martial arts demonstration a man with two swords faces a woman with a Chinese spear while another woman watches.
text_B: The woman is watching.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither? Is &quot;entailment&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · counterfactually-augmented-snli-283b54ff0e:train:3235

**State**

<pre>text_A: A child has fun with a knife in the kitchen.
text_B: A child has fun with appliances in the kitchen.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>
