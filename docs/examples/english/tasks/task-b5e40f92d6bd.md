# WANLI

Classification · `alisawuffles/WANLI`

<details><summary>Example 1</summary>

published Jev · train · WANLI-fc7ea2501e:train:pack-93da6f907962:label-A

**State**

<pre>Item A:
text_A: I think I will write about my dream.
text_B: I think I will write about my favorite color.

Item B:
text_A: What do you think of that?
text_B: You are telling me that you don&#x27;t think of it at all?

Item C:
text_A: The language of the new generation of film directors is not a new language, but a language that has been around for a long time.
text_B: The language of the new generation of film directors is a new language.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · WANLI-fc7ea2501e:train:pack-93da6f907962:same-B-C

**State**

<pre>Item A:
text_A: I think I will write about my dream.
text_B: I think I will write about my favorite color.

Item B:
text_A: What do you think of that?
text_B: You are telling me that you don&#x27;t think of it at all?

Item C:
text_A: The language of the new generation of film directors is not a new language, but a language that has been around for a long time.
text_B: The language of the new generation of film directors is a new language.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Do Item B and Item C have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · WANLI-fc7ea2501e:train:pack-93da6f907962:exists-0

**State**

<pre>Item A:
text_A: I think I will write about my dream.
text_B: I think I will write about my favorite color.

Item B:
text_A: What do you think of that?
text_B: You are telling me that you don&#x27;t think of it at all?

Item C:
text_A: The language of the new generation of film directors is not a new language, but a language that has been around for a long time.
text_B: The language of the new generation of film directors is a new language.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Does at least one item have the label &quot;entailment&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
