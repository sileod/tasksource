# anli/a3

Classification · `facebook/anli`

<details><summary>Example 1</summary>

published Jev · train · anli-a3-9d2e281b32:train:76

**State**

<pre>text_A: I got a card -- but you may be very sure I did n&#x27;t go , although Nancy thought I was crazy not to .
Then every one else gave parties in honor of Mr. Fenwick and I was invited and never went .
Wilhelmina Mercer came and pleaded and scolded and told me if I avoided Mr. Fenwick like that he would think I still cherished bitterness against him , and he would n&#x27;t make any advances towards a reconciliation .
Wilhelmina means well , but she has n&#x27;t a great deal of sense .
Cecil Fenwick seemed to be a great favorite with everybody , young and old .
text_B: Mercer was made at person of subject</pre>

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

published Jev · train · anli-a3-9d2e281b32:train:76:noul-label-verification

**State**

<pre>text_A: I got a card -- but you may be very sure I did n&#x27;t go , although Nancy thought I was crazy not to .
Then every one else gave parties in honor of Mr. Fenwick and I was invited and never went .
Wilhelmina Mercer came and pleaded and scolded and told me if I avoided Mr. Fenwick like that he would think I still cherished bitterness against him , and he would n&#x27;t make any advances towards a reconciliation .
Wilhelmina means well , but she has n&#x27;t a great deal of sense .
Cecil Fenwick seemed to be a great favorite with everybody , young and old .
text_B: Mercer was made at person of subject</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither? Is &quot;entailment&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · anli-a3-9d2e281b32:train:76:choice-criteria-permutation

**State**

<pre>text_A: I got a card -- but you may be very sure I did n&#x27;t go , although Nancy thought I was crazy not to .
Then every one else gave parties in honor of Mr. Fenwick and I was invited and never went .
Wilhelmina Mercer came and pleaded and scolded and told me if I avoided Mr. Fenwick like that he would think I still cherished bitterness against him , and he would n&#x27;t make any advances towards a reconciliation .
Wilhelmina means well , but she has n&#x27;t a great deal of sense .
Cecil Fenwick seemed to be a great favorite with everybody , young and old .
text_B: Mercer was made at person of subject</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. contradiction
2. neutral
3. entailment</pre>

**Gold target (weight)**

<pre>3. entailment — 1</pre>

</details>
