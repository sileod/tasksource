# it-support-tickets

Classification · `tasksource/it-support-tickets`

<details><summary>Example 1</summary>

published Jev · train · it-support-tickets-3a0c274acf:train:2

**State**

<pre>Remove File Share Access - [TICKET ID] - [NAME] ([COMPANY]A. [LOCATION])</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. Active Directory
2. Computer-Services
3. EOL
4. Fileservice
5. O365
6. Software
7. Support general</pre>

**Gold target (weight)**

<pre>4. Fileservice — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · it-support-tickets-3a0c274acf:train:2:noul-label-verification

**State**

<pre>Remove File Share Access - [TICKET ID] - [NAME] ([COMPANY]A. [LOCATION])</pre>

**Question**

<pre>Is &quot;EOL&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · it-support-tickets-3a0c274acf:train:1343

**State**

<pre>instalação power bi PRO</pre>

**Question**

<pre>Choose the criterion that best describes the state.</pre>

**Criteria (choice)**

<pre>1. Active Directory
2. Computer-Services
3. EOL
4. Fileservice
5. O365
6. Software
7. Support general</pre>

**Gold target (weight)**

<pre>5. O365 — 1</pre>

</details>
