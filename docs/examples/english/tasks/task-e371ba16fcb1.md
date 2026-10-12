# babi_nli/single-supporting-fact

Classification · `tasksource/babi_nli`

<details><summary>Example 1</summary>

published Jev · train · babi-nli-single-supporting-fact-64e2c2137a:train:3

**State**

<pre>text_A: John went to the bathroom. John went back to the hallway. Sandra went back to the kitchen. Daniel went back to the office.
text_B: John is in the hallway.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · babi-nli-single-supporting-fact-64e2c2137a:train:323

**State**

<pre>text_A: Mary went back to the office. John went back to the bathroom. John went to the hallway. Mary journeyed to the hallway. Sandra travelled to the hallway. Mary travelled to the office. Mary went back to the hallway. Sandra travelled to the office.
text_B: Sandra is in the office.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · babi-nli-single-supporting-fact-64e2c2137a:train:119

**State**

<pre>text_A: Sandra travelled to the office. Sandra went to the bathroom. Mary went to the bedroom. Daniel moved to the hallway. John went to the garden. John travelled to the office. Daniel journeyed to the bedroom. Daniel travelled to the hallway. John went to the bedroom. John travelled to the office.
text_B: Daniel is in the hallway.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. not-entailed
2. entailed</pre>

**Gold target (weight)**

<pre>2. entailed — 1</pre>

</details>
