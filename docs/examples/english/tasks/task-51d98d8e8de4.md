# probability_words_nli/reasoning_2hop

Classification · `sileod/probability_words_nli`

<details><summary>Example 1</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:0

**State**

<pre>There is a better than even chance that Gertrude is a sheep. It is highly unlikely that Lily is yellow. There is a better than even chance that Julius is a frog. There is a very good chance that if &#x27;Julius is a frog and Lily is yellow&#x27; then Bernhard is gray. We doubt that if either &#x27;Gertrude is a sheep&#x27; or &#x27;Julius is a frog&#x27; but not both then Brian is yellow. It is almost certain that if &#x27;Julius is a frog and Gertrude is a sheep&#x27; then John went to the garden.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
There is a better than even chance that &#x27;Bernhard is gray&#x27; or &#x27;Gertrude is a sheep&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 2</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:1

**State**

<pre>It is probably the case that Julius is green. We believe that Lily is gray. Chances are slight that Jeff discarded the milk. It is probably the case that if &#x27;Jeff discarded the milk&#x27; or &#x27;Julius is green&#x27; or both then Greg is a swan. Chances are slight that if &#x27;Lily is gray and Julius is green&#x27; then John moved to the garden. It is highly unlikely that if either &#x27;Jeff discarded the milk&#x27; or &#x27;Julius is green&#x27; but not both then Mary left the milk.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is improbable that &#x27;Mary left the milk&#x27; or &#x27;Jeff discarded the milk&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>

<details><summary>Example 3</summary>

current local Jev recast (not yet published) · train · probability-words-nli-reasoning-2hop-e10bbfbc39:train:2

**State**

<pre>It is highly likely that Sandra is in the kitchen. It is probably the case that Winona is a wolf. It is improbable that Greg is white. It is highly likely that if &#x27;Sandra is in the kitchen&#x27; or &#x27;Winona is a wolf&#x27; or both then Bill moved to the office. We believe that if &#x27;Greg is white and Sandra is in the kitchen&#x27; then Fred left the football. It is highly unlikely that if &#x27;Sandra is in the kitchen&#x27; or &#x27;Greg is white&#x27; or both then Bernhard is a rhino.</pre>

**Question**

<pre>Given the context, is the following conclusion valid?
It is probably the case that &#x27;Fred left the football&#x27; or &#x27;Winona is a wolf&#x27; or both.</pre>

**Criteria (choice)**

<pre>1. invalid
2. valid</pre>

**Gold target (weight)**

<pre>2. valid — 1</pre>

</details>
