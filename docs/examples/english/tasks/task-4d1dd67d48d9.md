# babi_nli/counting

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-counting-e9dfb27722:train:0

**State**

<pre>text_A: Daniel moved to the bathroom. John moved to the kitchen. Daniel went back to the kitchen. Sandra went to the hallway. Sandra went back to the bathroom. Sandra got the apple there.
text_B: There is one  objects is Sandra carrying.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-counting-e9dfb27722:train:115

**State**

<pre>text_A: Sandra journeyed to the kitchen. Sandra got the apple there. Daniel grabbed the football there. Daniel put down the football. Sandra journeyed to the bathroom. Daniel grabbed the football there. Mary travelled to the bedroom. John went to the bedroom.
text_B: There is one  objects is Daniel carrying.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-counting-e9dfb27722:train:109

**State**

<pre>text_A: Mary journeyed to the office. Mary went to the hallway. Daniel journeyed to the garden. John journeyed to the bathroom. Daniel travelled to the bedroom. John grabbed the apple there. John put down the apple. John got the apple there. John discarded the apple. Daniel took the milk there. Mary moved to the kitchen. Daniel dropped the milk.
text_B: There are none  objects is John carrying.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
