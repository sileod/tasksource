# openbookqa

MultipleChoice · `allenai/openbookqa`

<details><summary>Example 1</summary>

published Jev · train · openbookqa-51283af114:train:5

**State**

<pre>Poison causes harm to which of the following?</pre>

**Question**

<pre>Which supplied option best answers the question?</pre>

**Criteria (choice)**

<pre>1. a robot
2. a car
3. a Tree
4. a house</pre>

**Gold target (weight)**

<pre>3. a Tree — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · openbookqa-51283af114:train:5:noul-label-verification

**State**

<pre>Poison causes harm to which of the following?</pre>

**Question**

<pre>Is &quot;a Tree&quot; the correct answer to the question?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · openbookqa-51283af114:train:5:choice-criteria-permutation

**State**

<pre>Poison causes harm to which of the following?</pre>

**Question**

<pre>Which supplied option best answers the question?</pre>

**Criteria (choice)**

<pre>1. a Tree
2. a car
3. a house
4. a robot</pre>

**Gold target (weight)**

<pre>1. a Tree — 1</pre>

</details>
