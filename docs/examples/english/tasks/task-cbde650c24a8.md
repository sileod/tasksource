# babi_nli/two-arg-relations

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-two-arg-relations-9fc54432c3:train:2

**State**

<pre>text_A: The garden is west of the kitchen. The hallway is east of the kitchen.
text_B: The kitchen west of is kitchen.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-two-arg-relations-9fc54432c3:train:2:noul-label-verification

**State**

<pre>text_A: The garden is west of the kitchen. The hallway is east of the kitchen.
text_B: The kitchen west of is kitchen.</pre>

**Question**

<pre>Does text_A entail text_B? Is &quot;entailed&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-two-arg-relations-9fc54432c3:train:pack-99b5b116c0a8:label-A

**State**

<pre>Item A:
text_A: The kitchen is west of the hallway. The hallway is west of the bathroom.
text_B: The bathroom east of is kitchen.

Item B:
text_A: The kitchen is south of the garden. The office is north of the garden.
text_B: South of the garden is kitchen.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>
