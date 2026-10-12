# probability_words_nli/reasoning_1hop

Classification · `sileod/probability_words_nli`

<details><summary>Example 1</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-1hop-4241424585:train:0

**State**

<pre>It is likely that Sumit is thirsty. It is almost certain that John moved to the office. There is a very good chance that Mary left the apple.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is probably the case that &#x27;John moved to the office and Sumit is thirsty&#x27;.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 2</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-1hop-4241424585:train:1

**State**

<pre>It is highly likely that Julius is gray. It is impossible that Jeff went to the bedroom. It is likely that Lily is white.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is highly likely that &#x27;Julius is gray&#x27; or &#x27;Jeff went to the bedroom&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 3</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-1hop-4241424585:train:2

**State**

<pre>It is probable that Gertrude is a mouse. There is a very good chance that Bill went to the garden. It is probably not the case that Daniel dropped the milk.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
Chances are about even that either &#x27;Bill went to the garden&#x27; or &#x27;Gertrude is a mouse&#x27; but not both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>
