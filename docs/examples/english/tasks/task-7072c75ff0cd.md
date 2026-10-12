# wikisql/operator

Classification · `Salesforce/wikisql`

<details><summary>Example 1</summary>

published Jev · train · wikisql:train:2-15399928-3:What's the date for HOF week?:0:wikisql-operator-33e9e2c613

**State**

<pre>text_A: Schema: [[&quot;Week&quot;, &quot;text&quot;], [&quot;Date&quot;, &quot;text&quot;], [&quot;Opponent&quot;, &quot;text&quot;], [&quot;Result&quot;, &quot;text&quot;], [&quot;Game site&quot;, &quot;text&quot;], [&quot;Record&quot;, &quot;text&quot;], [&quot;Attendance&quot;, &quot;real&quot;]]
Question: What&#x27;s the date for HOF week?
text_B: Condition on column 0: Week; comparison value: hof</pre>

**Question**

<pre>Which comparison operator does the specified SQL condition require?</pre>

**Criteria (choice)**

<pre>1. equals
2. greater than
3. less than</pre>

**Gold target (weight)**

<pre>1. equals — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · wikisql:train:2-1140073-1:What was the engine belonging to Pierluigi Martini with a Tyre of P?:0:wikisql-operator-33e9e2c613

**State**

<pre>text_A: Schema: [[&quot;Entrant&quot;, &quot;text&quot;], [&quot;Constructor&quot;, &quot;text&quot;], [&quot;Chassis&quot;, &quot;text&quot;], [&quot;Engine&quot;, &quot;text&quot;], [&quot;Tyres&quot;, &quot;text&quot;], [&quot;Driver&quot;, &quot;text&quot;], [&quot;Rounds&quot;, &quot;text&quot;]]
Question: What was the engine belonging to Pierluigi Martini with a Tyre of P?
text_B: Condition on column 4: Tyres; comparison value: p</pre>

**Question**

<pre>Which comparison operator does the specified SQL condition require?</pre>

**Criteria (choice)**

<pre>1. equals
2. greater than
3. less than</pre>

**Gold target (weight)**

<pre>1. equals — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · wikisql:train:2-1434986-13:Which District has a First elected of 1858, and a Result of retired republican gain, and an Incumbent of william millward?:0:wikisql-operator-33e9e2c613

**State**

<pre>text_A: Schema: [[&quot;District&quot;, &quot;text&quot;], [&quot;Incumbent&quot;, &quot;text&quot;], [&quot;Party&quot;, &quot;text&quot;], [&quot;First elected&quot;, &quot;real&quot;], [&quot;Result&quot;, &quot;text&quot;]]
Question: Which District has a First elected of 1858, and a Result of retired republican gain, and an Incumbent of william millward?
text_B: Condition on column 3: First elected; comparison value: 1858</pre>

**Question**

<pre>Which comparison operator does the specified SQL condition require?</pre>

**Criteria (choice)**

<pre>1. equals
2. greater than
3. less than</pre>

**Gold target (weight)**

<pre>1. equals — 1</pre>

</details>
