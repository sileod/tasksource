# hh-rlhf/helpful-base

MultipleChoice · `tasksource/hh-rlhf`

<details><summary>Example 1</summary>

published Jev · train · hh-rlhf-helpful-base-cd92336a3b:train:90

**State**

<pre>Human: what is the closest planet to the sun</pre>

**Question**

<pre>Which next assistant reply is more helpful?</pre>

**Criteria (choice)**

<pre>1. Ganymede, one of the moons of Jupiter
2. (in) the Sun&#x27;s Heliocentric ecliptic:

1. Mercury    23.9 AU
2. Venus     67.2 AU
3. Earth      92.9 AU
4. Mars       142.</pre>

**Gold target (weight)**

<pre>2. (in) the Sun&#x27;s Heliocentric ecliptic:

1. Mercury    23.9 AU
2. Venus     67.2 AU
3. Earth      92.9 AU
4. Mars       142. — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · hh-rlhf-helpful-base-cd92336a3b:train:90:noul-label-verification

**State**

<pre>Human: what is the closest planet to the sun</pre>

**Question**

<pre>Which next assistant reply is more helpful? Is &quot;Ganymede, one of the moons of Jupiter&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · hh-rlhf-helpful-base-cd92336a3b:train:23308

**State**

<pre>Human: Can you list different variants of apples?</pre>

**Question**

<pre>Which next assistant reply is more helpful?</pre>

**Criteria (choice)**

<pre>1. Of course. Which ones do you want to start with? I’ve got a brief overview in the pdf at the link below. I could start with something generic, and go
2. Sure, I can do that. I also can tell you about the nutritional value of different kinds of apples, and provide some dietary tips.</pre>

**Gold target (weight)**

<pre>2. Sure, I can do that. I also can tell you about the nutritional value of different kinds of apples, and provide some dietary tips. — 1</pre>

</details>
