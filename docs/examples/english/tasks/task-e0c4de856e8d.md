# silicone/sem

Classification · `tasksource/silicone`

<details><summary>Example 1</summary>

published Jev · train · silicone-sem-a8f3a16c15:train:2855

**State**

<pre>-lrb- in quite voice , smiling -rrb- tomeo .</pre>

**Question**

<pre>What sentiment does the text express?</pre>

**Criteria (score)**

<pre>1. Negative
2. Neutral
3. Positive</pre>

**Gold target (weight)**

<pre>2. Neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · silicone-sem-a8f3a16c15:train:2448

**State**

<pre>hi ! how are you today ?</pre>

**Question**

<pre>What sentiment does the text express?</pre>

**Criteria (choice)**

<pre>1. Negative
2. Neutral
3. Positive</pre>

**Gold target (weight)**

<pre>2. Neutral — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · silicone-sem-a8f3a16c15:train:2448:noul-label-verification

**State**

<pre>hi ! how are you today ?</pre>

**Question**

<pre>What sentiment does the text express? Is &quot;Negative&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
