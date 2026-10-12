# super_glue/cb

Classification · `aps/super_glue`

<details><summary>Example 1</summary>

published Jev · train · super-glue-cb-37fdf25ac5:train:58

**State**

<pre>text_A: But he ended up eating it himself. I was reluctant to kiss my mother, afraid that somehow her weakness and unhappiness would infect me. Naturally I didn&#x27;t think for a minute that my life and spirit could stimulate her.
text_B: her life and spirit could stimulate her mother</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>2. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · super-glue-cb-37fdf25ac5:train:85

**State**

<pre>text_A: Your honesty shines out of your face, my darling. It isn&#x27;t your fault that cynical men like myself won&#x27;t let themselves believe what they see! I just wish you could believe that Eddie&#x27;s death was an accident and nothing to do with me.
text_B: Eddie&#x27;s death was an accident and nothing to do with him</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · super-glue-cb-37fdf25ac5:train:163

**State**

<pre>text_A: A: Have you followed that very much or, B: Uh, not really. I don&#x27;t think anything will ever take over the NFL.
text_B: something will take over the NFL</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>2. contradiction — 1</pre>

</details>
