# glue/qqp

Classification · `nyu-mll/glue`

<details><summary>Example 1</summary>

published Jev · train · glue-qqp-ae62fd302e:train:97

**State**

<pre>text_A: How can I solve rubix cube? Is there any formula for it?
text_B: How do I solve the magic cube?</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. not_duplicate
2. duplicate</pre>

**Gold target (weight)**

<pre>2. duplicate — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · glue-qqp-ae62fd302e:train:25012

**State**

<pre>Passage A:
How does corn and vegetable oil differ?

Passage B:
What&#x27;s the difference between corn and vegetable oil?</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. not_duplicate
2. duplicate</pre>

**Gold target (weight)**

<pre>2. duplicate — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · glue-qqp-ae62fd302e:train:25012:choice-criteria-permutation

**State**

<pre>Passage A:
How does corn and vegetable oil differ?

Passage B:
What&#x27;s the difference between corn and vegetable oil?</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. duplicate
2. not_duplicate</pre>

**Gold target (weight)**

<pre>1. duplicate — 1</pre>

</details>
