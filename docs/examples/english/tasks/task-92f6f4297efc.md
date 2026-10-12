# naturallogic

Classification · `tasksource/naturallogic`

<details><summary>Example 1</summary>

published Jev · train · naturallogic-ba86bebf3b:train:15

**State**

<pre>text_A: more than five of campers have caught a cold.
text_B: more than five of *people* have had a sunburn or caught a cold</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · naturallogic-ba86bebf3b:train:15:choice-criteria-permutation

**State**

<pre>text_A: more than five of campers have caught a cold.
text_B: more than five of *people* have had a sunburn or caught a cold</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>3. neutral — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · naturallogic-ba86bebf3b:train:2419

**State**

<pre>text_A: a californian policeman pulled a *plane* over and spoke to the driver
text_B: a californian policeman riding a motorcycle pulled a car over and spoke to the driver.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>
