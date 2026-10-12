# mindgames

Classification · `sileod/mindgames`

<details><summary>Example 1</summary>

published Jev · train · mindgames-dfe4fff9e3:train:96

**State**

<pre>text_A: There are four persons. Everyone is visible to others. It is publicly announced that someone is thirsty. It is publicly announced that Stephanie knows whether or not nobody is thirsty. It is publicly announced that Stephanie does not know whether Tracey is thirsty.
text_B: Stephanie can now know whether Vickie can know whether Stephanie is thirsty.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · mindgames-dfe4fff9e3:train:96:choice-paired-text-format

**State**

<pre>First text:
There are four persons. Everyone is visible to others. It is publicly announced that someone is thirsty. It is publicly announced that Stephanie knows whether or not nobody is thirsty. It is publicly announced that Stephanie does not know whether Tracey is thirsty.

Second text:
Stephanie can now know whether Vickie can know whether Stephanie is thirsty.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · mindgames-dfe4fff9e3:train:2131

**State**

<pre>text_A: There are four persons. Everyone is visible to others. It is publicly announced that someone is thirsty. It is publicly announced that Lawrence does not know whether Hazel is thirsty.
text_B: Elmer can now know that Elmer is thirsty.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>
