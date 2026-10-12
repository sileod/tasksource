# nope

Classification · `tasksource/nope`

<details><summary>Example 1</summary>

published Jev · train · nope-76272dc4fa:train:10

**State**

<pre>text_A:  Sports fell to the wayside, though, after he enrolled in a playwriting class at Occidental College in Los Angeles. He left in 1996 after starring with Owen and Andrew in Wes Anderson&#x27;s slackernoir classic, Bottle Rocket.
text_B: He starred with Owen and Andrew in Wes Anderson&#x27;s slackernoir classic, Bottle Rocket.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · nope-76272dc4fa:train:10:choice-paired-text-format

**State**

<pre>Passage A:
 Sports fell to the wayside, though, after he enrolled in a playwriting class at Occidental College in Los Angeles. He left in 1996 after starring with Owen and Andrew in Wes Anderson&#x27;s slackernoir classic, Bottle Rocket.

Passage B:
He starred with Owen and Andrew in Wes Anderson&#x27;s slackernoir classic, Bottle Rocket.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · nope-76272dc4fa:train:64

**State**

<pre>text_A: Settlements and legal fees have cost the church close to $2 billion. Six dioceses have entered bankruptcy. Many Catholics, observers say, remain skeptical about church leadership.
text_B: Many Catholics were already skeptical about church leadership.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
