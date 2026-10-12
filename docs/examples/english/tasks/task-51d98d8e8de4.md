# probability_words_nli/reasoning_2hop

Classification · `sileod/probability_words_nli`

<details><summary>Example 1</summary>

published Jev · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:2158

**State**

<pre>It is unlikely that Brian is a lion. It is unlikely that Bernhard is a rhino. It is probably the case that Daniel left the football. There is a very good chance that if either &#x27;Bernhard is a rhino&#x27; or &#x27;Daniel left the football&#x27; but not both then Mary went to the bedroom. We doubt that if &#x27;Bernhard is a rhino&#x27; or &#x27;Daniel left the football&#x27; or both then Julius is white. It is probably not the case that if &#x27;Brian is a lion and Bernhard is a rhino&#x27; then John moved to the office.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is likely that &#x27;John moved to the office&#x27; or &#x27;Daniel left the football&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:2235

**State**

<pre>There is little chance that Greg is a swan. It is improbable that Mary dropped the milk. It is almost certain that Jessica is a mouse. It is likely that if either &#x27;Jessica is a mouse&#x27; or &#x27;Greg is a swan&#x27; but not both then Emily is a sheep. We doubt that if either &#x27;Jessica is a mouse&#x27; or &#x27;Greg is a swan&#x27; but not both then Sandra took the football. It is highly unlikely that if &#x27;Jessica is a mouse&#x27; or &#x27;Greg is a swan&#x27; or both then Julius is a rhino.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
We doubt that either &#x27;Mary dropped the milk&#x27; or &#x27;Greg is a swan&#x27; but not both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:1790

**State**

<pre>There is a very good chance that Brian is white. It is highly likely that Mary got the apple. It is probably not the case that Greg is a swan. It is highly unlikely that if either &#x27;Mary got the apple&#x27; or &#x27;Greg is a swan&#x27; but not both then Jason is tired. It is certain that if &#x27;Mary got the apple&#x27; or &#x27;Brian is white&#x27; or both then Bernhard is yellow. It is probable that if either &#x27;Brian is white&#x27; or &#x27;Mary got the apple&#x27; but not both then John moved to the garden.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is likely that &#x27;Jason is tired and Mary got the apple&#x27;.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>1. invalid — 1</pre>

</details>
