# recast/recast_verbnet

Classification · `tasksource/recast`

<details><summary>Example 1</summary>

published Jev · train · recast-recast-verbnet-56e971bc66:train:1054

**State**

<pre>text_A: Tony broke the window with a hammer .
text_B: A hammer caused the breaking .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-recast-verbnet-56e971bc66:train:1054:choice-paired-text-format

**State**

<pre>Passage A:
Tony broke the window with a hammer .

Passage B:
A hammer caused the breaking .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-recast-verbnet-56e971bc66:train:962

**State**

<pre>text_A: Tony bent the rod .
text_B: The bending changed the physical form of Tony .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
