# gen_debiased_nli/snli_seq_z

Classification · `pietrolesci/gen_debiased_nli`

<details><summary>Example 1</summary>

published Jev · train · gen-debiased-nli-snli-seq-z-c01ef24799:train:pack-d557132d4173:label-B

**State**

<pre>Item A:
text_A: Two workers in hard hats examine a piece of industrial equipment in a tunnel underground.
text_B: Three workers in hard hats examine a piece of industrial equipment in a tunnel underground.

Item B:
text_A: Paul McCartney and John Lennon are performing on stage at Carnegie Hall in New York City.
text_B: Paul McCartney and John Lennon are taking photos in front of the Eiffel Tower.

Item C:
text_A: American footballer in yellow is pushed off the ball by his opponent in white.
text_B: A football player runs down the field after the home side has kicked the ball.

Item D:
text_A: Authorities work on the scene of a road-rage accident between a silver car and a black car, with many onlookers.
text_B: People sitting around a table</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · gen-debiased-nli-snli-seq-z-c01ef24799:train:pack-d557132d4173:same-B-D

**State**

<pre>Item A:
text_A: Two workers in hard hats examine a piece of industrial equipment in a tunnel underground.
text_B: Three workers in hard hats examine a piece of industrial equipment in a tunnel underground.

Item B:
text_A: Paul McCartney and John Lennon are performing on stage at Carnegie Hall in New York City.
text_B: Paul McCartney and John Lennon are taking photos in front of the Eiffel Tower.

Item C:
text_A: American footballer in yellow is pushed off the ball by his opponent in white.
text_B: A football player runs down the field after the home side has kicked the ball.

Item D:
text_A: Authorities work on the scene of a road-rage accident between a silver car and a black car, with many onlookers.
text_B: People sitting around a table</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Do Item B and Item D have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · gen-debiased-nli-snli-seq-z-c01ef24799:train:pack-d557132d4173:exists-1

**State**

<pre>Item A:
text_A: Two workers in hard hats examine a piece of industrial equipment in a tunnel underground.
text_B: Three workers in hard hats examine a piece of industrial equipment in a tunnel underground.

Item B:
text_A: Paul McCartney and John Lennon are performing on stage at Carnegie Hall in New York City.
text_B: Paul McCartney and John Lennon are taking photos in front of the Eiffel Tower.

Item C:
text_A: American footballer in yellow is pushed off the ball by his opponent in white.
text_B: A football player runs down the field after the home side has kicked the ball.

Item D:
text_A: Authorities work on the scene of a road-rage accident between a silver car and a black car, with many onlookers.
text_B: People sitting around a table</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Does at least one item have the label &quot;neutral&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
