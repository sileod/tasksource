# hans

Classification · `tasksource/hans`

<details><summary>Example 1</summary>

published Jev · train · hans-8800578b51:train:312

**State**

<pre>text_A: The lawyers mentioned the athlete .
text_B: The athlete mentioned the lawyers .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. non-entailment</pre>

**Gold target (weight)**

<pre>2. non-entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · hans-8800578b51:train:312:choice-paired-text-format

**State**

<pre>Passage A:
The lawyers mentioned the athlete .

Passage B:
The athlete mentioned the lawyers .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. non-entailment</pre>

**Gold target (weight)**

<pre>2. non-entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · hans-8800578b51:train:15000

**State**

<pre>text_A: The artist and the senator advised the author .
text_B: The senator advised the author .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. non-entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
