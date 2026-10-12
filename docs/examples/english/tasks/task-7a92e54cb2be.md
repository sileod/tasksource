# sentfin

Classification · `tasksource/sentfin`

<details><summary>Example 1</summary>

published Jev · train · sentfin:train:5058:Reliance Infrastructure:sentfin-a665621803

**State**

<pre>text_A: Brokers&#x27; call on  Exide Industries, Bharti Airtel, Reliance Infrastructure
text_B: Reliance Infrastructure</pre>

**Question**

<pre>What sentiment does headline text_A express toward entity text_B?</pre>

**Criteria (choice)**

<pre>1. negative
2. neutral
3. positive</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · sentfin:train:6870:NTPC:sentfin-a665621803

**State**

<pre>text_A: 191-192 is where price of NTPC to come in next few trading sessions: Sandeep Wagle
text_B: NTPC</pre>

**Question**

<pre>What sentiment does headline text_A express toward entity text_B?</pre>

**Criteria (score)**

<pre>1. negative
2. neutral
3. positive</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · sentfin:train:8685:two-wheeler:sentfin-a665621803

**State**

<pre>text_A: Cautious on two-wheeler space: David Pezarkar
text_B: two-wheeler</pre>

**Question**

<pre>What sentiment does headline text_A express toward entity text_B?</pre>

**Criteria (score)**

<pre>1. negative
2. neutral
3. positive</pre>

**Gold target (weight)**

<pre>1. negative — 1</pre>

</details>
