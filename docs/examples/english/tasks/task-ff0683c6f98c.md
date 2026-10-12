# lexical_relation_classification/K&H+N

Classification · `json`

<details><summary>Example 1</summary>

published Jev · train · lexical-relation-classification-K-H-N-7adfb9ec3a:train:pack-8ef9da39a4d6:label-A

**State**

<pre>Item A:
text_A: car
text_B: grille

Item B:
text_A: car
text_B: bonnet

Item C:
text_A: mollusc
text_B: shell</pre>

**Question**

<pre>Each item answers: &quot;How is the second word related to the first?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. co-hyponym
2. hyponym
3. meronym
4. unrelated</pre>

**Gold target (weight)**

<pre>3. meronym — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · lexical-relation-classification-K-H-N-7adfb9ec3a:train:pack-8ef9da39a4d6:same-B-C

**State**

<pre>Item A:
text_A: car
text_B: grille

Item B:
text_A: car
text_B: bonnet

Item C:
text_A: mollusc
text_B: shell</pre>

**Question**

<pre>Each item answers: &quot;How is the second word related to the first?&quot;
Do Item B and Item C have the same label? Possible labels: &quot;co-hyponym&quot;, &quot;hyponym&quot;, &quot;meronym&quot;, &quot;unrelated&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · lexical-relation-classification-K-H-N-7adfb9ec3a:train:pack-8ef9da39a4d6:exists-1

**State**

<pre>Item A:
text_A: car
text_B: grille

Item B:
text_A: car
text_B: bonnet

Item C:
text_A: mollusc
text_B: shell</pre>

**Question**

<pre>Each item answers: &quot;How is the second word related to the first?&quot;
Does at least one item have the label &quot;hyponym&quot;? Possible labels: &quot;co-hyponym&quot;, &quot;hyponym&quot;, &quot;meronym&quot;, &quot;unrelated&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
