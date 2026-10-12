# wikisql/column

MultipleChoice · `Salesforce/wikisql`

<details><summary>Example 1</summary>

published Jev · train · wikisql:train:2-15601335-1:What is the Round when there is an overall of 246?:wikisql-column-bacf0c140e

**State**

<pre>Schema: [[&quot;Round&quot;, &quot;real&quot;], [&quot;Overall&quot;, &quot;real&quot;], [&quot;Player&quot;, &quot;text&quot;], [&quot;Position&quot;, &quot;text&quot;], [&quot;College&quot;, &quot;text&quot;]]
Question: What is the Round when there is an overall of 246?</pre>

**Question**

<pre>Which column should the SQL query select to answer the question?</pre>

**Criteria (choice)**

<pre>1. Column 0: Round
2. Column 2: Player
3. Column 3: Position
4. Column 1: Overall
5. Column 4: College</pre>

**Gold target (weight)**

<pre>1. Column 0: Round — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · wikisql:train:1-20745754-1:What was the game #7, at or versus (home or at)?:wikisql-column-bacf0c140e

**State**

<pre>Schema: [[&quot;#&quot;, &quot;text&quot;], [&quot;Date&quot;, &quot;text&quot;], [&quot;at/vs.&quot;, &quot;text&quot;], [&quot;Opponent&quot;, &quot;text&quot;], [&quot;Score&quot;, &quot;text&quot;], [&quot;Attendance&quot;, &quot;real&quot;], [&quot;Record&quot;, &quot;text&quot;]]
Question: What was the game #7, at or versus (home or at)?</pre>

**Question**

<pre>Which column should the SQL query select to answer the question?</pre>

**Criteria (choice)**

<pre>1. Column 1: Date
2. Column 2: at/vs.
3. Column 4: Score
4. Column 5: Attendance
5. Column 0: #
6. Column 6: Record
7. Column 3: Opponent</pre>

**Gold target (weight)**

<pre>2. Column 2: at/vs. — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · wikisql:train:2-17100961-61:what is the pick when the overall is less than 297 and the college is alabama?:wikisql-column-bacf0c140e

**State**

<pre>Schema: [[&quot;Round&quot;, &quot;real&quot;], [&quot;Pick&quot;, &quot;real&quot;], [&quot;Overall&quot;, &quot;real&quot;], [&quot;Name&quot;, &quot;text&quot;], [&quot;Position&quot;, &quot;text&quot;], [&quot;College&quot;, &quot;text&quot;]]
Question: what is the pick when the overall is less than 297 and the college is alabama?</pre>

**Question**

<pre>Which column should the SQL query select to answer the question?</pre>

**Criteria (choice)**

<pre>1. Column 2: Overall
2. Column 1: Pick
3. Column 5: College
4. Column 0: Round
5. Column 3: Name
6. Column 4: Position</pre>

**Gold target (weight)**

<pre>2. Column 1: Pick — 1</pre>

</details>
