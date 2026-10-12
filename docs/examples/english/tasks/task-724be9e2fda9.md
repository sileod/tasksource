# multidogo/airline/intent

Classification · `jpcorb20/multidogo`

<details><summary>Example 1</summary>

published Jev · train · multidogo:train:<CONV>dad66e07-94c3-4e0c-be13-688c2d1e88f1<TURN>2<SENT>0:multidogo-airline-intent-b02020c0b4

**State**

<pre>text_A: Turn 0: hai
text_B: i need to check my seat assignment</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. bookflight
2. changeseatassignment
3. closinggreeting
4. confirmation
5. contentonly
6. getboardingpass
7. getseatinfo
8. openinggreeting
9. outofdomain
10. rejection
11. thankyou</pre>

**Gold target (weight)**

<pre>2. changeseatassignment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · multidogo:train:<CONV>d40f040d-aac1-4ab9-973c-42853ddc982e<TURN>6<SENT>0:multidogo-airline-intent-b02020c0b4

**State**

<pre>text_A: Turn 0: hai
Turn 4: kumar 968 ukl
text_B: yes please continue</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. bookflight
2. changeseatassignment
3. closinggreeting
4. confirmation
5. contentonly
6. getboardingpass
7. getseatinfo
8. openinggreeting
9. outofdomain
10. rejection
11. thankyou</pre>

**Gold target (weight)**

<pre>4. confirmation — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · multidogo:train:<CONV>71048671-e044-4092-9a8c-ecb119ee33b2<TURN>1<SENT>1:multidogo-airline-intent-b02020c0b4

**State**

<pre>text_A: Turn 1: hi
text_B: id like to get my boarding pass sent to me please</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. bookflight
2. changeseatassignment
3. closinggreeting
4. confirmation
5. contentonly
6. getboardingpass
7. getseatinfo
8. openinggreeting
9. outofdomain
10. rejection
11. thankyou</pre>

**Gold target (weight)**

<pre>6. getboardingpass — 1</pre>

</details>
