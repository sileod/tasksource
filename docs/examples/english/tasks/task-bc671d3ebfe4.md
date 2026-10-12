# vitaminc

Classification · `tals/vitaminc`

<details><summary>Example 1</summary>

published Jev · train · vitaminc-ad0b6a3bdd:train:18177

**State**

<pre>Passage A:
David Alaba played for the Austrian senior national team when he was 18 years old .

Passage B:
He holds Austria &#x27;s record as the youngest player to play for their senior national team , debuting for them in 2009 as a 18-year-old .</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. NOT ENOUGH INFO
2. REFUTES
3. SUPPORTS</pre>

**Gold target (weight)**

<pre>3. SUPPORTS — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · vitaminc-ad0b6a3bdd:train:2872

**State**

<pre>text_A: James has a son and a daughter..
text_B: James is married to Michelle and has two children .</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. NOT ENOUGH INFO
2. REFUTES
3. SUPPORTS</pre>

**Gold target (weight)**

<pre>1. NOT ENOUGH INFO — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · vitaminc-ad0b6a3bdd:train:11939

**State**

<pre>text_A: MLS extended once again its intended return date indefinitely on April 14 , 2020 .
text_B: On 19 March , Major League Soccer &#x27;s suspension was extended to a target return date of 10 May . As of April 14 , MLS has announced that it will not return on the intended date of May 10 and will continue to analyze other options .</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. NOT ENOUGH INFO
2. REFUTES
3. SUPPORTS</pre>

**Gold target (weight)**

<pre>3. SUPPORTS — 1</pre>

</details>
