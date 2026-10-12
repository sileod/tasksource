# glue/qnli

Classification · `nyu-mll/glue`

<details><summary>Example 1</summary>

published Jev · train · glue-qnli-2a1193e085:train:844

**State**

<pre>text_A: What language did the Huns adopt?
text_B: After Attila&#x27;s unexpected death the Hunnic Empire collapsed with the Huns disappearing as a people in Europe – who either escaped into Asia, or otherwise blended in amongst Europeans.</pre>

**Question**

<pre>Does sentence text_B contain the answer to question text_A?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · glue-qnli-2a1193e085:train:844:noul-label-verification

**State**

<pre>text_A: What language did the Huns adopt?
text_B: After Attila&#x27;s unexpected death the Hunnic Empire collapsed with the Huns disappearing as a people in Europe – who either escaped into Asia, or otherwise blended in amongst Europeans.</pre>

**Question**

<pre>Does sentence text_B contain the answer to question text_A? Is &quot;not_entailment&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · glue-qnli-2a1193e085:train:11157

**State**

<pre>text_A: Who won the 1980 US Presidential Elections after President Carter&#x27;s repeated failed attempts to resolve the US Iranian Embassy situation?
text_B: On November 4, 1979, a group of students seized the United States Embassy and took the embassy with 52 personnel and citizens hostage, after the United States refused to return Mohammad Reza Pahlavi to Iran to face trial in the court of the new regime.</pre>

**Question**

<pre>Does sentence text_B contain the answer to question text_A?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>
