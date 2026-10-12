# zero-shot-label-nli

Classification · `tasksource/zero-shot-label-nli`

<details><summary>Example 1</summary>

current local Jev recast (not yet published) · train · zero-shot-label-nli-eb601fa5e1:train:0

**State**

<pre>text_A: Jarrah was intimately familiar with , and adept at using the intercom and radio . 
Jarrah , like Atta earlier , may have inadvertently broadcast the message because he did not know how to operate the radio and the intercom .
text_B: This example is non-contradiction.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

current local Jev recast (not yet published) · train · zero-shot-label-nli-eb601fa5e1:train:1

**State**

<pre>text_A: @user Am Eye See Kay He Why Am Oh You Ass He
text_B: This example is non-offensive.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 3</summary>

current local Jev recast (not yet published) · train · zero-shot-label-nli-eb601fa5e1:train:2

**State**

<pre>text_A: elm
wood
text_B: This example is attri.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>
