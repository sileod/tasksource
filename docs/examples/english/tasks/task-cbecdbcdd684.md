# lingnli

Classification · `tasksource/lingnli`

<details><summary>Example 1</summary>

published Jev · train · lingnli-36f6080abb:train:pack-dfe03029e28a:label-A

**State**

<pre>Item A:
text_A: The charities, Paris adds, don&#x27;t endorse the product at all.
text_B: The charities thoroughly endorse all of the products.

Item B:
text_A: Time describes honor killings in Jordan, which comprise a quarter of the Arab nation&#x27;s homicides.
text_B: The honor killings in Jordan are based on religious beliefs.

Item C:
text_A: Thanks to misinformation unrivaled since the pre-FDA patent-medicine era, however, alternative therapies have become a $14 billion market, which is growing by as much as 30 percent annually.
text_B: Misinformation will continue and will see market growth continue.</pre>

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

published Jev · train · lingnli-36f6080abb:train:pack-dfe03029e28a:same-A-C

**State**

<pre>Item A:
text_A: The charities, Paris adds, don&#x27;t endorse the product at all.
text_B: The charities thoroughly endorse all of the products.

Item B:
text_A: Time describes honor killings in Jordan, which comprise a quarter of the Arab nation&#x27;s homicides.
text_B: The honor killings in Jordan are based on religious beliefs.

Item C:
text_A: Thanks to misinformation unrivaled since the pre-FDA patent-medicine era, however, alternative therapies have become a $14 billion market, which is growing by as much as 30 percent annually.
text_B: Misinformation will continue and will see market growth continue.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Do Item A and Item C have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · lingnli-36f6080abb:train:pack-dfe03029e28a:exists-1

**State**

<pre>Item A:
text_A: The charities, Paris adds, don&#x27;t endorse the product at all.
text_B: The charities thoroughly endorse all of the products.

Item B:
text_A: Time describes honor killings in Jordan, which comprise a quarter of the Arab nation&#x27;s homicides.
text_B: The honor killings in Jordan are based on religious beliefs.

Item C:
text_A: Thanks to misinformation unrivaled since the pre-FDA patent-medicine era, however, alternative therapies have become a $14 billion market, which is growing by as much as 30 percent annually.
text_B: Misinformation will continue and will see market growth continue.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Does at least one item have the label &quot;neutral&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
