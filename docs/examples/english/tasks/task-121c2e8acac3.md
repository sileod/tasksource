# help-nli

Classification · `tasksource/help-nli`

<details><summary>Example 1</summary>

published Jev · train · help-nli-63b76526b1:train:pack-6801b484b991:label-B

**State**

<pre>Item A:
text_A: GM has initiated criminal investigations in Germany
text_B: GM has initiated criminal investigations in Germany and the US on the basis of their suspicions that Mr Lopez and his team stole confidential information before defecting to VW in March .

Item B:
text_A: There were no radios in those days .
text_B: There were no superheterodyne receivers in those days .</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · help-nli-63b76526b1:train:pack-6801b484b991:same-A-B

**State**

<pre>Item A:
text_A: GM has initiated criminal investigations in Germany
text_B: GM has initiated criminal investigations in Germany and the US on the basis of their suspicions that Mr Lopez and his team stole confidential information before defecting to VW in March .

Item B:
text_A: There were no radios in those days .
text_B: There were no superheterodyne receivers in those days .</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Do Item A and Item B have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · help-nli-63b76526b1:train:pack-6801b484b991:exists-1

**State**

<pre>Item A:
text_A: GM has initiated criminal investigations in Germany
text_B: GM has initiated criminal investigations in Germany and the US on the basis of their suspicions that Mr Lopez and his team stole confidential information before defecting to VW in March .

Item B:
text_A: There were no radios in those days .
text_B: There were no superheterodyne receivers in those days .</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Does at least one item have the label &quot;neutral&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
