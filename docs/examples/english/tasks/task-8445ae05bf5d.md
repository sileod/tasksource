# gen_debiased_nli/mnli_seq_z

Classification · `pietrolesci/gen_debiased_nli`

<details><summary>Example 1</summary>

published Jev · train · gen-debiased-nli-mnli-seq-z-126d32fedb:train:113

**State**

<pre>text_A: The main entrance takes you to the first sacred gateway (torii), which in Japan always symbolizes the threshold of holy ground.
text_B: There are three Toriis that link the holy grounds together.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · gen-debiased-nli-mnli-seq-z-126d32fedb:train:113:choice-paired-text-format

**State**

<pre>First text:
The main entrance takes you to the first sacred gateway (torii), which in Japan always symbolizes the threshold of holy ground.

Second text:
There are three Toriis that link the holy grounds together.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · gen-debiased-nli-mnli-seq-z-126d32fedb:train:24806

**State**

<pre>text_A: &#x27;So what&#x27;re you going to do?&#x27; I demanded.
text_B: You need to make a decision.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>
