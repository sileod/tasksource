# lonli

Classification · `tasksource/lonli`

<details><summary>Example 1</summary>

published Jev · train · lonli-c26dcc495d:train:61

**State**

<pre>text_A: Joseph has 9 dollars. He received 7 more dollars.
text_B: Joseph now has 2 dollars.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · lonli-c26dcc495d:train:61:choice-criteria-permutation

**State**

<pre>text_A: Joseph has 9 dollars. He received 7 more dollars.
text_B: Joseph now has 2 dollars.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>2. contradiction — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · lonli-c26dcc495d:train:12265

**State**

<pre>text_A: Harry has geography class then computer science class then music class.
text_B: Geography class is the last class among the three.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>
