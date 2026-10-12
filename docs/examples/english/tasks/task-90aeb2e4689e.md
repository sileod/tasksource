# wikisql/aggregation

Classification · `Salesforce/wikisql`

<details><summary>Example 1</summary>

published Jev · train · wikisql:train:2-15601335-1:What is the Round when there is an overall of 246?:wikisql-aggregation-e497c0575a

**State**

<pre>Schema: [[&quot;Round&quot;, &quot;real&quot;], [&quot;Overall&quot;, &quot;real&quot;], [&quot;Player&quot;, &quot;text&quot;], [&quot;Position&quot;, &quot;text&quot;], [&quot;College&quot;, &quot;text&quot;]]
Question: What is the Round when there is an overall of 246?</pre>

**Question**

<pre>What SQL aggregation should be applied to answer the question?</pre>

**Criteria (choice)**

<pre>1. none
2. maximum
3. minimum
4. count
5. sum
6. average</pre>

**Gold target (weight)**

<pre>1. none — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · wikisql:train:1-20745754-1:What was the game #7, at or versus (home or at)?:wikisql-aggregation-e497c0575a

**State**

<pre>Schema: [[&quot;#&quot;, &quot;text&quot;], [&quot;Date&quot;, &quot;text&quot;], [&quot;at/vs.&quot;, &quot;text&quot;], [&quot;Opponent&quot;, &quot;text&quot;], [&quot;Score&quot;, &quot;text&quot;], [&quot;Attendance&quot;, &quot;real&quot;], [&quot;Record&quot;, &quot;text&quot;]]
Question: What was the game #7, at or versus (home or at)?</pre>

**Question**

<pre>What SQL aggregation should be applied to answer the question?</pre>

**Criteria (choice)**

<pre>1. none
2. maximum
3. minimum
4. count
5. sum
6. average</pre>

**Gold target (weight)**

<pre>1. none — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · wikisql:train:2-17100961-61:what is the pick when the overall is less than 297 and the college is alabama?:wikisql-aggregation-e497c0575a

**State**

<pre>Schema: [[&quot;Round&quot;, &quot;real&quot;], [&quot;Pick&quot;, &quot;real&quot;], [&quot;Overall&quot;, &quot;real&quot;], [&quot;Name&quot;, &quot;text&quot;], [&quot;Position&quot;, &quot;text&quot;], [&quot;College&quot;, &quot;text&quot;]]
Question: what is the pick when the overall is less than 297 and the college is alabama?</pre>

**Question**

<pre>What SQL aggregation should be applied to answer the question?</pre>

**Criteria (choice)**

<pre>1. none
2. maximum
3. minimum
4. count
5. sum
6. average</pre>

**Gold target (weight)**

<pre>1. none — 1</pre>

</details>
