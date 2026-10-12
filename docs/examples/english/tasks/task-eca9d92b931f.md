# joci

Classification · `pietrolesci/joci`

<details><summary>Example 1</summary>

published Jev · train · joci-47f0b47b28:train:360

**State**

<pre>Passage A:
A band that is playing outside for a benefit in New York .

Passage B:
A person improves the benefit .</pre>

**Question**

<pre>How likely is the hypothesis, given the context?</pre>

**Criteria (choice)**

<pre>1. impossible
2. technically possible
3. plausible
4. likely
5. very likely</pre>

**Gold target (weight)**

<pre>3. plausible — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · joci-47f0b47b28:train:360:noul-label-verification

**State**

<pre>Passage A:
A band that is playing outside for a benefit in New York .

Passage B:
A person improves the benefit .</pre>

**Question**

<pre>How likely is the hypothesis, given the context? Is &quot;very likely&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · joci-47f0b47b28:train:360:choice-criteria-permutation

**State**

<pre>Passage A:
A band that is playing outside for a benefit in New York .

Passage B:
A person improves the benefit .</pre>

**Question**

<pre>How likely is the hypothesis, given the context?</pre>

**Criteria (choice)**

<pre>1. impossible
2. technically possible
3. likely
4. very likely
5. plausible</pre>

**Gold target (weight)**

<pre>5. plausible — 1</pre>

</details>
