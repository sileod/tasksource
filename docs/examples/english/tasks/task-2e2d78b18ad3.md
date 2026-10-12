# sick/relatedness

Classification · `tasksource/sick`

<details><summary>Example 1</summary>

published Jev · train · tasksource-sick-9e0850a4d0:train:5bd6a12a5df18ac6:sick-relatedness

**State**

<pre>text_A: A man on a horse is riding on a track and waving
text_B: A man on a motorcycle is riding on a track and waving</pre>

**Question**

<pre>How related in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. 1 (completely unrelated)
2. 2
3. 3
4. 4
5. 5 (very related)</pre>

**Gold target (weight)**

<pre>2. 2 — 0.1
3. 3 — 0.9</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · tasksource-sick-9e0850a4d0:train:e2dfa9bb4ed2b2fe:sick-relatedness

**State**

<pre>First text:
Two people in snowsuits are lying in the snow and making snow angels

Second text:
Two angels are making snow on the lying children</pre>

**Question**

<pre>How related in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. 1 (completely unrelated)
2. 2
3. 3
4. 4
5. 5 (very related)</pre>

**Gold target (weight)**

<pre>2. 2 — 0.5
3. 3 — 0.5</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · tasksource-sick-9e0850a4d0:train:5b733d069d6b8a05:sick-relatedness

**State**

<pre>A: A girl is skipping rope in a park
B: A girl is skipping rope on a sidewalk</pre>

**Question**

<pre>How related in meaning are the two sentences?</pre>

**Criteria (score)**

<pre>1. 1 (completely unrelated)
2. 2
3. 3
4. 4
5. 5 (very related)</pre>

**Gold target (weight)**

<pre>4. 4 — 1</pre>

</details>
