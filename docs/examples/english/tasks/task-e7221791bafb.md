# stepgame

Classification · `tasksource/stepgame`

<details><summary>Example 1</summary>

published Jev · train · stepgame-1cc74264d9:train:29785

**State**

<pre>A: Y is over there and C is on the left of it. D is below Y with a small gap between them. The object labeled C is positioned to the left of the object labeled M.
B: What is the relation of the agent Y to the agent M?</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. above
2. below
3. left
4. lower-left
5. lower-right
6. overlap
7. right
8. upper-left
9. upper-right</pre>

**Gold target (weight)**

<pre>6. overlap — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · stepgame-1cc74264d9:train:10436

**State**

<pre>First text:
U is on the right and C is on the left. A is diagonally to the bottom left of J. A and W are parallel, and A is to the right of W. O is to the top-left of C. K is diagonally right and below I. J is positioned above O and to the left. I is above H at 2 o&#x27;clock. N is above and to the left of U. W and H are both there with the object W is to the left of object H.

Second text:
What is the relation of the agent A to the agent O?</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. above
2. below
3. left
4. lower-left
5. lower-right
6. overlap
7. right
8. upper-left
9. upper-right</pre>

**Gold target (weight)**

<pre>3. left — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · stepgame-1cc74264d9:train:16857

**State**

<pre>Passage A:
U and T are both there with the object U is to the left of object T. U is at a 45 degree angle to B, in the lower lefthand corner. S is over there and O is on the right of it. X is directly below S. J is upper left of C. C is sitting at the 12:00 position to T. J presents over X.

Passage B:
What is the relation of the agent T to the agent B?</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. above
2. below
3. left
4. lower-left
5. lower-right
6. overlap
7. right
8. upper-left
9. upper-right</pre>

**Gold target (weight)**

<pre>2. below — 1</pre>

</details>
