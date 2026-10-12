# glue/stsb

Classification · `nyu-mll/glue`

<details><summary>Example 1</summary>

published Jev · train · nyu-mll-glue-stsb-ad720c97de:train:7dd20420b943e21e:glue-stsb

**State**

<pre>First text:
A man is playing a large flute.

Second text:
A man is playing a flute.</pre>

**Question**

<pre>How similar in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. completely dissimilar
2. not equivalent, but on the same topic
3. not equivalent, but share some details
4. roughly equivalent, but some important information differs
5. mostly equivalent, but some unimportant details differ
6. completely equivalent</pre>

**Gold target (weight)**

<pre>4. roughly equivalent, but some important information differs — 0.2
5. mostly equivalent, but some unimportant details differ — 0.8</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · nyu-mll-glue-stsb-ad720c97de:train:17b404ad7bfca2ed:glue-stsb

**State**

<pre>text_A: Car Bombs Kill 58 in Iraqi Capital
text_B: Car bombings kill 13 civilians in Iraqi capital</pre>

**Question**

<pre>How similar in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. completely dissimilar
2. not equivalent, but on the same topic
3. not equivalent, but share some details
4. roughly equivalent, but some important information differs
5. mostly equivalent, but some unimportant details differ
6. completely equivalent</pre>

**Gold target (weight)**

<pre>3. not equivalent, but share some details — 0.6
4. roughly equivalent, but some important information differs — 0.4</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · nyu-mll-glue-stsb-ad720c97de:train:b410cb60093fecc8:glue-stsb

**State**

<pre>text_A: Two dogs on the beach.
text_B: A dog walks on the beach.</pre>

**Question**

<pre>How similar in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. completely dissimilar
2. not equivalent, but on the same topic
3. not equivalent, but share some details
4. roughly equivalent, but some important information differs
5. mostly equivalent, but some unimportant details differ
6. completely equivalent</pre>

**Gold target (weight)**

<pre>3. not equivalent, but share some details — 0.6
4. roughly equivalent, but some important information differs — 0.4</pre>

</details>
