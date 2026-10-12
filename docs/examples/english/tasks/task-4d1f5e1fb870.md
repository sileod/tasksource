# regset

Classification · `tasksource/regset`

<details><summary>Example 1</summary>

published Jev · train · regset-68fd053640:train:15672

**State**

<pre>b(a|((aa)*b)*) baabaabaaaab</pre>

**Question**

<pre>Does the string match the regular expression?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>2. True — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · regset-68fd053640:train:21477

**State**

<pre>(b((aa)*|ab))* bbbaaa</pre>

**Question**

<pre>Does the string match the regular expression?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · regset-68fd053640:train:21477:noul-label-verification

**State**

<pre>(b((aa)*|ab))* bbbaaa</pre>

**Question**

<pre>Does the string match the regular expression? Is &quot;True&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
