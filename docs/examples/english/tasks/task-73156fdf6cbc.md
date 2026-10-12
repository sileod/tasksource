# recast_white/dpr

Classification · `pietrolesci/recast_white`

<details><summary>Example 1</summary>

published Jev · train · recast-white-dpr-428ec197ab:train:3015

**State**

<pre>text_A: Ken scanned photos for Neil since he owned a scanner .
text_B: Ken scanned photos for Neil since Neil owned a scanner .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-white-dpr-428ec197ab:train:3015:choice-criteria-permutation

**State**

<pre>text_A: Ken scanned photos for Neil since he owned a scanner .
text_B: Ken scanned photos for Neil since Neil owned a scanner .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailed
2. not-entailed</pre>

**Gold target (weight)**

<pre>2. not-entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-white-dpr-428ec197ab:train:3015:choice-paired-text-format

**State**

<pre>Passage A:
Ken scanned photos for Neil since he owned a scanner .

Passage B:
Ken scanned photos for Neil since Neil owned a scanner .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
