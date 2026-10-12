# Politeness_Disagreement

Classification · `RuyuanWan/Politeness_Disagreement`

<details><summary>Example 1</summary>

published Jev · train · Politeness-Disagreement-8e63cd8514:train:0

**State**

<pre>In your nom of Everyking you wrote &#x27;&#x27;...as Everyking is a former administrator, he made over 1,300 in his 2+ year time as a sysop.&#x27;&#x27; Should that be over 1,300 sysop actions?</pre>

**Question**

<pre>Would annotators disagree about the politeness of this text?</pre>

**Criteria (choice)**

<pre>1. annotators agree
2. annotators disagree</pre>

**Gold target (weight)**

<pre>1. annotators agree — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · Politeness-Disagreement-8e63cd8514:train:pack-0ace4545efe2:label-A

**State**

<pre>Item A:
&quot;Malbrain, I can&#x27;t understand you, in either your comments to the talk page...&quot; is the criteria being used by user WHOSASKING to make reverts of my edits.  Since when does any one editor have to understand all the material of a subject known by the other editors?

Item B:
The consequences apparently have no sliding scale? It&#x27;s either &quot;you&#x27;re blocked,&quot; or you&#x27;re not?

Item C:
So you see KV, just in case you missed the point, you are in the naughty chair in the naughty corner because of your own baloney, baloney that you bring yourself.  Now, didn&#x27;t you like that story?</pre>

**Question**

<pre>Each item answers: &quot;Would annotators disagree about the politeness of this text?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. annotators agree
2. annotators disagree</pre>

**Gold target (weight)**

<pre>1. annotators agree — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · Politeness-Disagreement-8e63cd8514:train:pack-0ace4545efe2:same-B-C

**State**

<pre>Item A:
&quot;Malbrain, I can&#x27;t understand you, in either your comments to the talk page...&quot; is the criteria being used by user WHOSASKING to make reverts of my edits.  Since when does any one editor have to understand all the material of a subject known by the other editors?

Item B:
The consequences apparently have no sliding scale? It&#x27;s either &quot;you&#x27;re blocked,&quot; or you&#x27;re not?

Item C:
So you see KV, just in case you missed the point, you are in the naughty chair in the naughty corner because of your own baloney, baloney that you bring yourself.  Now, didn&#x27;t you like that story?</pre>

**Question**

<pre>Each item answers: &quot;Would annotators disagree about the politeness of this text?&quot;
Do Item B and Item C have the same label? Possible labels: &quot;annotators agree&quot;, &quot;annotators disagree&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
