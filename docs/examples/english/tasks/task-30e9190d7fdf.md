# liar

Classification · `tasksource/liar`

<details><summary>Example 1</summary>

published Jev · train · liar-61e652ad7c:train:421

**State**

<pre>Says although the unemployment rate seems to be improving, it does not reflect real job creation -- its caused by a shrinking of the actual labor force.</pre>

**Question**

<pre>How true is the statement, according to PolitiFact?</pre>

**Criteria (choice)**

<pre>1. pants on fire
2. false
3. barely true
4. half true
5. mostly true
6. true</pre>

**Gold target (weight)**

<pre>5. mostly true — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · liar-61e652ad7c:train:421:noul-label-verification

**State**

<pre>Says although the unemployment rate seems to be improving, it does not reflect real job creation -- its caused by a shrinking of the actual labor force.</pre>

**Question**

<pre>How true is the statement, according to PolitiFact? Is &quot;half true&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · liar-61e652ad7c:train:421:choice-criteria-permutation

**State**

<pre>Says although the unemployment rate seems to be improving, it does not reflect real job creation -- its caused by a shrinking of the actual labor force.</pre>

**Question**

<pre>How true is the statement, according to PolitiFact?</pre>

**Criteria (choice)**

<pre>1. true
2. mostly true
3. half true
4. false
5. pants on fire
6. barely true</pre>

**Gold target (weight)**

<pre>2. mostly true — 1</pre>

</details>
