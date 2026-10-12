# hotpotqa/supporting-fact

Classification · `hotpotqa/hotpot_qa`

<details><summary>Example 1</summary>

published Jev · train · hotpotqa:train:5ae1fb09554299234fd04382:The Formula (1980 film):2:hotpotqa-supporting-fact-274a376cc2

**State**

<pre>text_A: Who casted a film directed by John G. Avildsen?
text_B: Title: The Formula (1980 film)
Sentence:  Craig T. Nelson also makes a brief appearance as a geologist.</pre>

**Question**

<pre>Is candidate sentence text_B annotated as supporting evidence for question text_A?</pre>

**Criteria (choice)**

<pre>1. not a supporting fact
2. supporting fact</pre>

**Gold target (weight)**

<pre>1. not a supporting fact — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · hotpotqa:train:5a80feba554299260e20a1f3:Las Meninas (film):6:hotpotqa-supporting-fact-274a376cc2

**State**

<pre>text_A: Are Ihor Podolchak and Pamela B. Green both co-founders of Masoch Fund?
text_B: Title: Las Meninas (film)
Sentence:  In 2011, it was included in Top 15 Best Ukrainian films of the 20 years’ Independence period.</pre>

**Question**

<pre>Is candidate sentence text_B annotated as supporting evidence for question text_A?</pre>

**Criteria (choice)**

<pre>1. not a supporting fact
2. supporting fact</pre>

**Gold target (weight)**

<pre>1. not a supporting fact — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · hotpotqa:train:5a8675b15542994775f60772:Yeah 3x:0:hotpotqa-supporting-fact-274a376cc2

**State**

<pre>text_A: Europop included the lead single, released on what date?
text_B: Title: Yeah 3x
Sentence: &quot;Yeah 3x&quot; (pronounced &quot;Yeah three times&quot; or &quot;Yeah Yeah Yeah&quot;; sometimes stylized as &quot;Yeah 3X&quot;) is a song by American recording artist Chris Brown, released as the lead single from his fourth studio album &quot;F.A.M.E.&quot; on October 25, 2010.</pre>

**Question**

<pre>Is candidate sentence text_B annotated as supporting evidence for question text_A?</pre>

**Criteria (choice)**

<pre>1. not a supporting fact
2. supporting fact</pre>

**Gold target (weight)**

<pre>1. not a supporting fact — 1</pre>

</details>
