# multidogo/software/intent

Classification · `jpcorb20/multidogo`

<details><summary>Example 1</summary>

published Jev · train · multidogo:train:<CONV>8955f4e2-93f4-4d6c-9d63-f4eb3fb3ee73<TURN>15<SENT>0:multidogo-software-intent-7b749275b2

**State**

<pre>text_A: Turn 0: hai
Turn 2: i want to reimbursed hotel room
Turn 4: camile 5465
Turn 6: 120
Turn 8: okay
Turn 10: okay conform
Turn 11: yes
Turn 13: okay
text_B: thankyou</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. changeorder
2. checkserverstatus
3. closinggreeting
4. confirmation
5. contentonly
6. expensereport
7. getpromotions
8. openinggreeting
9. outofdomain
10. providereceipt
11. rejection
12. reportbrokensoftware
13. softwareupdate
14. startorder
15. stoporder
16. thankyou</pre>

**Gold target (weight)**

<pre>16. thankyou — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · multidogo:train:<CONV>82836b6f-7558-45de-9aae-9c2a53294630<TURN>13<SENT>0:multidogo-software-intent-7b749275b2

**State**

<pre>text_A: Turn 0: hi
Turn 2: checking the global status of servers
Turn 4: lilly
Turn 5: 6985
Turn 7: samsung
Turn 9: ok
Turn 11: ok
text_B: no</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. changeorder
2. checkserverstatus
3. closinggreeting
4. confirmation
5. contentonly
6. expensereport
7. getpromotions
8. openinggreeting
9. outofdomain
10. providereceipt
11. rejection
12. reportbrokensoftware
13. softwareupdate
14. startorder
15. stoporder
16. thankyou</pre>

**Gold target (weight)**

<pre>11. rejection — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · multidogo:train:<CONV>920087d2-0e1c-4335-8d15-7e8b141bff85<TURN>3<SENT>0:multidogo-software-intent-7b749275b2

**State**

<pre>text_A: Turn 0: hello
Turn 2: i need to file a claim for my flight last week
text_B: oklahoma city 573</pre>

**Question**

<pre>What intent does the current utterance in text_B express, given earlier dialogue text_A?</pre>

**Criteria (choice)**

<pre>1. changeorder
2. checkserverstatus
3. closinggreeting
4. confirmation
5. contentonly
6. expensereport
7. getpromotions
8. openinggreeting
9. outofdomain
10. providereceipt
11. rejection
12. reportbrokensoftware
13. softwareupdate
14. startorder
15. stoporder
16. thankyou</pre>

**Gold target (weight)**

<pre>5. contentonly — 1</pre>

</details>
