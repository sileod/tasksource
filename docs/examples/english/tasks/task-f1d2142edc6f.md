# FLUTE

Classification · `ColumbiaNLP/FLUTE`

<details><summary>Example 1</summary>

published Jev · train · FLUTE-bf1fbd31ef:train:pack-685c55131750:label-A

**State**

<pre>Item A:
text_A: If you disallow or forbid the photos, I&#x27;ll move on the dorchester.
text_B: If you give the go ahead on the photos, I&#x27;ll move on the dorchester.

Item B:
text_A: All the meadows are frozen and chilly.
text_B: All the meadows  wave  with blossoms ,</pre>

**Question**

<pre>Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. Contradiction
2. Entailment</pre>

**Gold target (weight)**

<pre>1. Contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · FLUTE-bf1fbd31ef:train:pack-685c55131750:same-A-B

**State**

<pre>Item A:
text_A: If you disallow or forbid the photos, I&#x27;ll move on the dorchester.
text_B: If you give the go ahead on the photos, I&#x27;ll move on the dorchester.

Item B:
text_A: All the meadows are frozen and chilly.
text_B: All the meadows  wave  with blossoms ,</pre>

**Question**

<pre>Do Item A and Item B have the same label? Possible labels: &quot;Contradiction&quot;, &quot;Entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · FLUTE-bf1fbd31ef:train:pack-685c55131750:exists-1

**State**

<pre>Item A:
text_A: If you disallow or forbid the photos, I&#x27;ll move on the dorchester.
text_B: If you give the go ahead on the photos, I&#x27;ll move on the dorchester.

Item B:
text_A: All the meadows are frozen and chilly.
text_B: All the meadows  wave  with blossoms ,</pre>

**Question**

<pre>Does at least one item have the label &quot;Entailment&quot;? Possible labels: &quot;Contradiction&quot;, &quot;Entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
