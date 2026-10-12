# circa

Classification · `google-research-datasets/circa`

<details><summary>Example 1</summary>

published Jev · train · circa-9b8c5093f3:train:1

**State**

<pre>First text:
Y has just told X that he/she is thinking of buying a flat in New York. Do you go to New York often?

Second text:
I have never been there</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. Yes
2. No
3. In the middle, neither yes nor no
4. Yes, subject to some conditions
5. Other</pre>

**Gold target (weight)**

<pre>2. No — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · circa-9b8c5093f3:train:13236

**State**

<pre>Passage A:
Y has just told X that he/she is thinking of buying a flat in New York. Do you need to be close to the subway?

Passage B:
I don&#x27;t think I do</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. Yes
2. No
3. In the middle, neither yes nor no
4. Yes, subject to some conditions
5. Other</pre>

**Gold target (weight)**

<pre>2. No — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · circa-9b8c5093f3:train:13236:noul-label-verification

**State**

<pre>Passage A:
Y has just told X that he/she is thinking of buying a flat in New York. Do you need to be close to the subway?

Passage B:
I don&#x27;t think I do</pre>

**Question**

<pre>Is &quot;No&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
