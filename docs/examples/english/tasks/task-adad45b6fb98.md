# tracie

Classification · `tasksource/tracie`

<details><summary>Example 1</summary>

published Jev · train · tracie-7cf6863641:train:822

**State**

<pre>text_A: Dina found an ad on the news for a cheap car. So Dina went to the dealership. She soon found out that the it was deceptive. The dealer only gave cheap cars for people with perfect credit. So Dina ditched the dealership.
text_B: Dina drove to the dealership starts after Dina found an ad for a cheap car</pre>

**Question**

<pre>Given the story in text_A, is the temporal relation stated in text_B likely true?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not-entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · tracie-7cf6863641:train:567

**State**

<pre>text_A: Jeremy brought his boombox to school. Everyone was very confused. He started playing it very loud. Teachers got angry and got security. They gave him suspension.
text_B: Jeremy played his boombox in the halls at school ends after teachers got security</pre>

**Question**

<pre>Given the story in text_A, is the temporal relation stated in text_B likely true?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not-entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · tracie-7cf6863641:train:602

**State**

<pre>text_A: Liza was a star student in her high school. She walked home from the library one evening. A man pulled her into the alley and started groping her. Liza pulled out her pepper spray but it malfunctioned. Some students noticed the attacker and he fled.
text_B: Liza felt depressed after the event. starts before she left the library</pre>

**Question**

<pre>Given the story in text_A, is the temporal relation stated in text_B likely true?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not-entailment</pre>

**Gold target (weight)**

<pre>2. not-entailment — 1</pre>

</details>
