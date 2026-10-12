# recast/recast_factuality

Classification · `tasksource/recast`

<details><summary>Example 1</summary>

published Jev · train · recast-recast-factuality-2d3b4f6f36:train:7319

**State**

<pre>text_A: The Nasdaq Composite lost 54.15 points , or 4 % , to 1299.59 , while the Standard &amp; Poor &#x27;s 500 plunged by 30.27 points , or 4.25 % , closing at 682.60 .
text_B: The losing did not happen</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · recast-recast-factuality-2d3b4f6f36:train:29671

**State**

<pre>text_A: Can not ask for a better experience .
text_B: The asking happened</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · recast-recast-factuality-2d3b4f6f36:train:22671

**State**

<pre>text_A: Officials said last week that six people were in custody for questioning about the bombing .
text_B: The saying happened</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
