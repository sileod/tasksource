# tweet_eval/offensive

Classification · `cardiffnlp/tweet_eval`

<details><summary>Example 1</summary>

published Jev · train · tweet-eval-offensive-3fb522105d:train:3328

**State**

<pre>@user I heard they actually are gonna cgi an Antifa flag Now. You know. For accuracy</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. non-offensive
2. offensive</pre>

**Gold target (weight)**

<pre>1. non-offensive — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · tweet-eval-offensive-3fb522105d:train:3328:noul-label-verification

**State**

<pre>@user I heard they actually are gonna cgi an Antifa flag Now. You know. For accuracy</pre>

**Question**

<pre>Is &quot;non-offensive&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · tweet-eval-offensive-3fb522105d:train:138

**State**

<pre>@user You are sexy as fuck babygirl🔥🔥🔥</pre>

**Question**

<pre>Choose the criterion that best describes the state.</pre>

**Criteria (choice)**

<pre>1. non-offensive
2. offensive</pre>

**Gold target (weight)**

<pre>2. offensive — 1</pre>

</details>
