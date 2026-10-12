# SDOH-NLI

Classification · `tasksource/SDOH-NLI`

<details><summary>Example 1</summary>

published Jev · train · SDOH-NLI-f61a63216f:train:43

**State**

<pre>text_A: 1-2 ppd Cigarettes
text_B: The person is employed part time.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · SDOH-NLI-f61a63216f:train:43:choice-paired-text-format

**State**

<pre>Passage A:
1-2 ppd Cigarettes

Passage B:
The person is employed part time.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · SDOH-NLI-f61a63216f:train:1286

**State**

<pre>text_A: Denies any IV drug use.
text_B: The person is a student.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>
