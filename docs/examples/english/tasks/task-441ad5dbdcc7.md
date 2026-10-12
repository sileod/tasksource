# paws/labeled_final

Classification · `google-research-datasets/paws`

<details><summary>Example 1</summary>

published Jev · train · paws-labeled-final-cfc482d72f:train:340

**State**

<pre>text_A: Waye played his earliest football in Willunga and Renmark before arriving in Adelaide and participating in the Port Adelaide Church Association .
text_B: Waye played his earliest football game in Adelaide before arriving in Willunga and Renmark and joining the Port Adelaide Church Association .</pre>

**Question**

<pre>Is text_B a paraphrase of text_A?</pre>

**Criteria (choice)**

<pre>1. not_paraphrase
2. paraphrase</pre>

**Gold target (weight)**

<pre>1. not_paraphrase — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · paws-labeled-final-cfc482d72f:train:340:noul-label-verification

**State**

<pre>text_A: Waye played his earliest football in Willunga and Renmark before arriving in Adelaide and participating in the Port Adelaide Church Association .
text_B: Waye played his earliest football game in Adelaide before arriving in Willunga and Renmark and joining the Port Adelaide Church Association .</pre>

**Question**

<pre>Is text_B a paraphrase of text_A? Is &quot;paraphrase&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · paws-labeled-final-cfc482d72f:train:26295

**State**

<pre>text_A: He replaced John Mabry as backup at 1st Base to Derrek Lee .
text_B: He replaced Derrek Lee as a backup at 1st base to John Mabry .</pre>

**Question**

<pre>Is text_B a paraphrase of text_A?</pre>

**Criteria (choice)**

<pre>1. not_paraphrase
2. paraphrase</pre>

**Gold target (weight)**

<pre>1. not_paraphrase — 1</pre>

</details>
