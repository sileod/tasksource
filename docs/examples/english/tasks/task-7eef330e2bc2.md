# recast_white/fnplus

Classification · `pietrolesci/recast_white`

<details><summary>Example 1</summary>

published Jev · train · recast-white-fnplus-b1f6304667:train:27785

**State**

<pre>text_A: my sheik , here &#x27;s some of the event that happened in ramadi :
text_B: my sheik , here &#x27;s some of the happenings that happened in ramadi :</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-white-fnplus-b1f6304667:train:27785:choice-paired-text-format

**State**

<pre>Passage A:
my sheik , here &#x27;s some of the event that happened in ramadi :

Passage B:
my sheik , here &#x27;s some of the happenings that happened in ramadi :</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-white-fnplus-b1f6304667:train:11078

**State**

<pre>text_A: but with his death the empire collapsed , and the israelite kingdom was divided into two separate , impoverished , often warring nations : israel , with its capital at shechem in the north , ruled by a series of northern dynasties ; and the smaller kingdom of judah , with its capital at jerusalem , from which the davidic dynasty continued to rule .
text_B: but with his death the empire collapsed , and the israelite kingdom was divided into two separate , impoverished , often warring nations : israel , with its capital at shechem in the north , determined by a series of northern dynasties ; and the smaller kingdom of judah , with its capital at jerusalem , from which the davidic dynasty continued to rule .</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
