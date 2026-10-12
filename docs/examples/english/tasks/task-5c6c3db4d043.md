# recast/recast_ner

Classification · `tasksource/recast`

<details><summary>Example 1</summary>

published Jev · train · recast-recast-ner-8c89ce5b92:train:21200

**State**

<pre>text_A: Mr. Zuma said the government chose to bid for the World Cup because it `` wanted to show that Africa could do it . ``
text_B: Zuma is a natural entity</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-recast-ner-8c89ce5b92:train:6761

**State**

<pre>text_A: The group said Israel also limits the Palestinian Authority &#x27;s ability to drill new wells .
text_B: Israel is a location</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-recast-ner-8c89ce5b92:train:6761:choice-criteria-permutation

**State**

<pre>text_A: The group said Israel also limits the Palestinian Authority &#x27;s ability to drill new wells .
text_B: Israel is a location</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailed
2. not-entailed</pre>

**Gold target (weight)**

<pre>1. entailed — 1</pre>

</details>
