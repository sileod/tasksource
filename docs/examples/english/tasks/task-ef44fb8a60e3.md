# babi_nli/basic-deduction

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-basic-deduction-2f54b7a691:train:2

**State**

<pre>text_A: Wolves are afraid of sheep. Mice are afraid of sheep. Cats are afraid of mice. Gertrude is a wolf. Jessica is a mouse. Sheep are afraid of mice. Winona is a wolf. Emily is a cat.
text_B: Winona is afraid of mouse.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-basic-deduction-2f54b7a691:train:44

**State**

<pre>text_A: Cats are afraid of mice. Wolves are afraid of mice. Gertrude is a wolf. Sheep are afraid of cats. Winona is a sheep. Mice are afraid of wolves. Emily is a wolf. Jessica is a cat.
text_B: Jessica is afraid of cat.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>1. not-entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-basic-deduction-2f54b7a691:train:38

**State**

<pre>text_A: Cats are afraid of wolves. Wolves are afraid of sheep. Mice are afraid of sheep. Sheep are afraid of cats. Emily is a mouse. Jessica is a wolf. Gertrude is a sheep. Winona is a cat.
text_B: Gertrude is afraid of cat.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
