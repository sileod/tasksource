# paws/labeled_swap

Classification · `google-research-datasets/paws`

<details><summary>Example 1</summary>

published Jev · train · paws-labeled-swap-1850de3e98:train:87

**State**

<pre>text_A: The music written by S.M.Subbaiah Naïdu , while lyrics composed by Vaali .
text_B: The music composed by S.M.Subbaiah Naïdu , while lyrics written by Vaali</pre>

**Question**

<pre>Is text_B a paraphrase of text_A?</pre>

**Criteria (choice)**

<pre>1. not_paraphrase
2. paraphrase</pre>

**Gold target (weight)**

<pre>1. not_paraphrase — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · paws-labeled-swap-1850de3e98:train:13087

**State**

<pre>text_A: The `` Berkeley &#x27;&#x27; album is influenced by punk and garage , while the `` Bakersfield &#x27;&#x27; album leans toward the band &#x27;s `` California country &#x27;&#x27; side .
text_B: The `` Berkeley &#x27;&#x27; album is influenced by punk and garage , while the `` California &#x27;&#x27; album leans toward the band &#x27;s `` Bakersfield country &#x27;&#x27; side .</pre>

**Question**

<pre>Is text_B a paraphrase of text_A?</pre>

**Criteria (choice)**

<pre>1. not_paraphrase
2. paraphrase</pre>

**Gold target (weight)**

<pre>1. not_paraphrase — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · paws-labeled-swap-1850de3e98:train:13087:noul-label-verification

**State**

<pre>text_A: The `` Berkeley &#x27;&#x27; album is influenced by punk and garage , while the `` Bakersfield &#x27;&#x27; album leans toward the band &#x27;s `` California country &#x27;&#x27; side .
text_B: The `` Berkeley &#x27;&#x27; album is influenced by punk and garage , while the `` California &#x27;&#x27; album leans toward the band &#x27;s `` Bakersfield country &#x27;&#x27; side .</pre>

**Question**

<pre>Is text_B a paraphrase of text_A? Is &quot;paraphrase&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
