# logical-entailment

Classification · `tasksource/logical-entailment`

<details><summary>Example 1</summary>

published Jev · train · logical-entailment-5fde239036:train:pack-98d7a485359a:label-A

**State**

<pre>Item A:
text_A: (y&amp;((y&gt;(~(x)|~((j&amp;y))))&amp;(j&amp;~((u&amp;j)))))
text_B: ((j&amp;u)&gt;~(~(v)))

Item B:
text_A: (~((s|(r&amp;q)))&amp;m)
text_B: (x&gt;~((q&amp;(c|c))))</pre>

**Question**

<pre>Each item answers: &quot;Does propositional formula text_A logically entail formula text_B?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. entailment
2. not entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · logical-entailment-5fde239036:train:pack-98d7a485359a:in-B-0

**State**

<pre>Item A:
text_A: (y&amp;((y&gt;(~(x)|~((j&amp;y))))&amp;(j&amp;~((u&amp;j)))))
text_B: ((j&amp;u)&gt;~(~(v)))

Item B:
text_A: (~((s|(r&amp;q)))&amp;m)
text_B: (x&gt;~((q&amp;(c|c))))</pre>

**Question**

<pre>Each item answers: &quot;Does propositional formula text_A logically entail formula text_B?&quot;
Is the label of Item B &quot;entailment&quot;? Possible labels: &quot;entailment&quot;, &quot;not entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · logical-entailment-5fde239036:train:pack-98d7a485359a:all-same

**State**

<pre>Item A:
text_A: (y&amp;((y&gt;(~(x)|~((j&amp;y))))&amp;(j&amp;~((u&amp;j)))))
text_B: ((j&amp;u)&gt;~(~(v)))

Item B:
text_A: (~((s|(r&amp;q)))&amp;m)
text_B: (x&gt;~((q&amp;(c|c))))</pre>

**Question**

<pre>Each item answers: &quot;Does propositional formula text_A logically entail formula text_B?&quot;
Do all items have the same label? Possible labels: &quot;entailment&quot;, &quot;not entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
