# SpaRTUN

MultipleChoice · `tasksource/SpaRTUN`

<details><summary>Example 1</summary>

published Jev · train · SpaRTUN-ae090fe669:train:31

**State**

<pre>There are three boxes, called DDD, EEE and JJJ. Disconnected from box JJJ and box EEE there are box DDD with a midsize green rectangle. Box JJJ is under this box. Above and behind box DDD there is box EEE. A midsize orange rectangle is in box DDD. Box DDD covers another midsize green rectangle. A midsize white rectangle is covered by box EEE. Is any midsize white rectangle disconnected from the midsize orange rectangle?</pre>

**Question**

<pre>Choose the criterion that best answers the question.</pre>

**Criteria (choice)**

<pre>1. Yes
2. No</pre>

**Gold target (weight)**

<pre>1. Yes — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · SpaRTUN-ae090fe669:train:31:noul-label-verification

**State**

<pre>There are three boxes, called DDD, EEE and JJJ. Disconnected from box JJJ and box EEE there are box DDD with a midsize green rectangle. Box JJJ is under this box. Above and behind box DDD there is box EEE. A midsize orange rectangle is in box DDD. Box DDD covers another midsize green rectangle. A midsize white rectangle is covered by box EEE. Is any midsize white rectangle disconnected from the midsize orange rectangle?</pre>

**Question**

<pre>Is &quot;No&quot; the correct answer to the question?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · SpaRTUN-ae090fe669:train:31:choice-criteria-permutation

**State**

<pre>There are three boxes, called DDD, EEE and JJJ. Disconnected from box JJJ and box EEE there are box DDD with a midsize green rectangle. Box JJJ is under this box. Above and behind box DDD there is box EEE. A midsize orange rectangle is in box DDD. Box DDD covers another midsize green rectangle. A midsize white rectangle is covered by box EEE. Is any midsize white rectangle disconnected from the midsize orange rectangle?</pre>

**Question**

<pre>Choose the criterion that best answers the question.</pre>

**Criteria (choice)**

<pre>1. No
2. Yes</pre>

**Gold target (weight)**

<pre>2. Yes — 1</pre>

</details>
