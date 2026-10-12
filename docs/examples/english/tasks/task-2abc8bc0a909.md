# tweets_hate_speech_detection

Classification · `tweets-hate-speech-detection/tweets_hate_speech_detection`

<details><summary>Example 1</summary>

published Jev · train · tweets-hate-speech-detection-8cd975265a:train:24320

**State**

<pre>ready to ride #frenchchampionship   #waves @user @user @user @user </pre>

**Question**

<pre>Choose the criterion that best describes the state.</pre>

**Criteria (choice)**

<pre>1. no-hate-speech
2. hate-speech</pre>

**Gold target (weight)**

<pre>1. no-hate-speech — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · tweets-hate-speech-detection-8cd975265a:train:24320:noul-label-verification

**State**

<pre>ready to ride #frenchchampionship   #waves @user @user @user @user </pre>

**Question**

<pre>Is &quot;hate-speech&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · tweets-hate-speech-detection-8cd975265a:train:6227

**State**

<pre>@user @user @user you made my day by this announcement...can&#x27;t wait to see you in la ☺ </pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. no-hate-speech
2. hate-speech</pre>

**Gold target (weight)**

<pre>1. no-hate-speech — 1</pre>

</details>
