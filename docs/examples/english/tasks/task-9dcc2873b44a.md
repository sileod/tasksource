# probability_words_nli/reasoning_1hop

Classification · `sileod/probability_words_nli`

<details><summary>Example 1</summary>

published Jev · train · probability-words-nli-reasoning-1hop-4241424585:train:3378

**State**

<pre>It is certain that Mary got the milk. There is a very good chance that Greg is a rhino. We doubt that Lily is white.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is probable that &#x27;Lily is white and Greg is a rhino&#x27;.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>1. invalid — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · probability-words-nli-reasoning-1hop-4241424585:train:2521

**State**

<pre>There is little chance that Bernhard is yellow. There is almost no chance that John dropped the milk. It is probably the case that Julius is a swan.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
We believe that &#x27;John dropped the milk&#x27; or &#x27;Bernhard is yellow&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>1. invalid — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · probability-words-nli-reasoning-1hop-4241424585:train:2431

**State**

<pre>Chances are slight that Daniel left the apple. There is a very good chance that Brian is yellow. It is highly unlikely that Mary moved to the garden.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is probably the case that &#x27;Mary moved to the garden and Brian is yellow&#x27;.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>1. invalid — 1</pre>

</details>
