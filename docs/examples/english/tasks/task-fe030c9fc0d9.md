# recast/recast_megaveridicality

Classification · `tasksource/recast`

<details><summary>Example 1</summary>

published Jev · train · recast-recast-megaveridicality-aabd0ae501:train:7412

**State**

<pre>text_A: a particular person was upheld to do a particular thing .
text_B: that person might or might not have done that thing .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-recast-megaveridicality-aabd0ae501:train:5888

**State**

<pre>text_A: a particular person was n&#x27;t named to do a particular thing .
text_B: that person might or might not have done that thing .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-recast-megaveridicality-aabd0ae501:train:4183

**State**

<pre>text_A: someone witnessed that a particular thing happened .
text_B: that thing didn&#x27;t happen .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
