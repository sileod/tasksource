# temporal-nli

Classification · `tasksource/temporal-nli`

<details><summary>Example 1</summary>

published Jev · train · temporal-nli-0d8184c0f1:train:16431

**State**

<pre>text_A: In Dec 1920, this article was published.
text_B: After Sep 1978, this article was published.</pre>

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

published Jev · train · temporal-nli-0d8184c0f1:train:19246

**State**

<pre>text_A: Jake is planning to move in 3 months.
text_B: After 9 days, Jake is planning to move.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · temporal-nli-0d8184c0f1:train:19246:choice-paired-text-format

**State**

<pre>A: Jake is planning to move in 3 months.
B: After 9 days, Jake is planning to move.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
