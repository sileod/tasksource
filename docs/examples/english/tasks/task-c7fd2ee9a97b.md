# monli

Classification · `tasksource/monli`

<details><summary>Example 1</summary>

published Jev · train · monli-e20b1da17a:train:38

**State**

<pre>text_A: The girl here is not wearing any necklaces at all.
text_B: The girl here is not wearing any jewelry at all.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · monli-e20b1da17a:train:38:choice-paired-text-format

**State**

<pre>A: The girl here is not wearing any necklaces at all.
B: The girl here is not wearing any jewelry at all.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · monli-e20b1da17a:train:96

**State**

<pre>text_A: There is not a single programmer walking in the city.
text_B: There is not a single person walking in the city.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>
