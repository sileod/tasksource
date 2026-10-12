# ambient

Classification · `tasksource/ambient`

<details><summary>Example 1</summary>

published Jev · train · ambient-9f8bc859e5:train:0

**State**

<pre>A: The novel has been translated into more than 40 languages and has sold over 30 million copies worldwide.
B: The movie has been translated into more than 40 languages and has grossed over $1 billion worldwide.</pre>

**Question**

<pre>Is the hypothesis ambiguous?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · ambient-9f8bc859e5:train:19

**State**

<pre>Passage A:
The cashier told me that I didn&#x27;t have enough money to buy that.

Passage B:
I didn&#x27;t have enough money to buy that.</pre>

**Question**

<pre>Is the hypothesis ambiguous?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · ambient-9f8bc859e5:train:20

**State**

<pre>text_A: Of course not.
text_B: Certainly not.</pre>

**Question**

<pre>Is the hypothesis ambiguous?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>
