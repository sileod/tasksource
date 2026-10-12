# spartqa-yn

Classification · `tasksource/spartqa-yn`

<details><summary>Example 1</summary>

published Jev · train · spartqa-yn-a9b30ae554:train:11697

**State**

<pre>First text:
There are three blocks. Lets call them A, B and C. Block B is to the left of C. Block C is to the left of A. Block A has one medium yellow square, one medium black square and one medium yellow triangle. To the left of the medium yellow triangle there is the medium yellow square. The medium black square is below the medium yellow triangle. It is touching the bottom edge of this block. The medium yellow square is above the medium black square. Block B has a medium yellow triangle and one medium yellow circle. The medium yellow triangle is to the left of and above the medium yellow circle. And block C contains a small blue circle, a big black circle and a big yellow triangle. The small blue circle is to the left of the big black circle which is to the right of the big yellow triangle. This object is near to the big yellow triangle.

Second text:
Is there a yellow shape that is to the left of a small blue circle , to the left of the medium black square that is below a medium yellow thing?</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. No
2. Yes
3. don&#x27;t know</pre>

**Gold target (weight)**

<pre>2. Yes — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · spartqa-yn-a9b30ae554:train:4522

**State**

<pre>First text:
There are three blocks. We call them A, B and C. Block A has a medium black square and one medium blue square. The medium black square is touching the bottom edge of this block. It is below the medium blue square. Above block A there is block B which has a medium yellow square and a medium blue square. Below the medium blue square there is the medium yellow square. Finally, we have block C to the left of block B. It contains one medium yellow square.

Second text:
Are all medium squares which are below a blue square , near to a medium shape that is below a medium square?</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. No
2. Yes
3. don&#x27;t know</pre>

**Gold target (weight)**

<pre>1. No — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · spartqa-yn-a9b30ae554:train:3801

**State**

<pre>First text:
We have three blocks. Lets call them A, B and C. Block C is below block A and above block B. Block A has one medium black square. Block B has a medium yellow square and one medium black square. The medium yellow square is touching the bottom edge of this block. It is below the medium black square. Block C has two medium blue squares. Medium blue square number one is below medium blue square number two and a medium black square. There is also a medium yellow square in this block. It is below medium blue square number two. Below medium blue square number two there is the medium black square. It is below the yellow shape which is below medium blue square number two.

Second text:
Are all medium yellow squares that are below a medium square , to the left of all black objects that are above a blue square?</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. No
2. Yes
3. don&#x27;t know</pre>

**Gold target (weight)**

<pre>3. don&#x27;t know — 1</pre>

</details>
