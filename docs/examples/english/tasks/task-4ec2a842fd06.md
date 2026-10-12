# humicroedit/subtask-1

Classification · `tasksource/humicroedit`

<details><summary>Example 1</summary>

published Jev · train · humicroedit-subtask-1-5903f8a67b:train:124

**State**

<pre>Passage A:
Original: North Korea missiles : US warships deployed to Korean peninsula - BBC News

Passage B:
Edited: North Korea missiles : US warships deployed to Korean saunas - BBC News</pre>

**Question**

<pre>How funny is the edited headline?</pre>

**Criteria (choice)**

<pre>1. not funny
2. slightly funny
3. moderately funny
4. funny</pre>

**Gold target (weight)**

<pre>2. slightly funny — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · humicroedit-subtask-1-5903f8a67b:train:124:noul-label-verification

**State**

<pre>Passage A:
Original: North Korea missiles : US warships deployed to Korean peninsula - BBC News

Passage B:
Edited: North Korea missiles : US warships deployed to Korean saunas - BBC News</pre>

**Question**

<pre>How funny is the edited headline? Is &quot;slightly funny&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · humicroedit-subtask-1-5903f8a67b:train:2941

**State**

<pre>text_A: Original: Maryam Mirzakhani , Only Woman to Win a Fields Medal , Dies at 40
text_B: Edited: Maryam Mirzakhani , Only landscaper to Win a Fields Medal , Dies at 40</pre>

**Question**

<pre>How funny is the edited headline?</pre>

**Criteria (choice)**

<pre>1. not funny
2. slightly funny
3. moderately funny
4. funny</pre>

**Gold target (weight)**

<pre>1. not funny — 1</pre>

</details>
