# clutrr

Classification · `tasksource/clutrr`

<details><summary>Example 1</summary>

published Jev · train · clutrr-effc7ced7a:train:pack-f59e25835dde:label-B

**State**

<pre>Item A:
text_A: Pedro made a pizza for his daughter. Her name is Rebecca. . One day Antonio the uncle of Rebecca, decided to surprise Rebecca with a camping trip. Rebecca had been wanting to go camping for a long time. Pedro took his daughter, Rebecca, to the father daughter dance at church. Antonio enjoys talking to his sister. Her name is Beverly.
text_B: How is Beverly related to Rebecca?

Item B:
text_A: Harold is taking his son Louis on a camping trip for the weekend. Harold loves his daughter Beverly. Louis is the great brother of Antonio. Antonio read a bedtime story to his daughter Emma.
text_B: How is Beverly related to Emma?

Item C:
text_A: Shantel made sandwiches with her daughters Dorothy and Marie. Marie &#x27;daughter Michelle made a salad. Beverly and her mother Shantel went to volunteer at the homeless shelter last weekend.
text_B: How is Beverly related to Michelle?</pre>

**Question**

<pre>Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. aunt
2. brother
3. daughter
4. daughter-in-law
5. father
6. father-in-law
7. granddaughter
8. grandfather
9. grandmother
10. grandson
11. mother
12. mother-in-law
13. nephew
14. niece
15. sister
16. son
17. son-in-law
18. uncle</pre>

**Gold target (weight)**

<pre>1. aunt — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · clutrr-effc7ced7a:train:pack-f59e25835dde:in-B-1.2.6.8.9.10.13.14

**State**

<pre>Item A:
text_A: Pedro made a pizza for his daughter. Her name is Rebecca. . One day Antonio the uncle of Rebecca, decided to surprise Rebecca with a camping trip. Rebecca had been wanting to go camping for a long time. Pedro took his daughter, Rebecca, to the father daughter dance at church. Antonio enjoys talking to his sister. Her name is Beverly.
text_B: How is Beverly related to Rebecca?

Item B:
text_A: Harold is taking his son Louis on a camping trip for the weekend. Harold loves his daughter Beverly. Louis is the great brother of Antonio. Antonio read a bedtime story to his daughter Emma.
text_B: How is Beverly related to Emma?

Item C:
text_A: Shantel made sandwiches with her daughters Dorothy and Marie. Marie &#x27;daughter Michelle made a salad. Beverly and her mother Shantel went to volunteer at the homeless shelter last weekend.
text_B: How is Beverly related to Michelle?</pre>

**Question**

<pre>Is the label of Item B one of &quot;brother&quot;, &quot;daughter&quot;, &quot;granddaughter&quot;, &quot;grandmother&quot;, &quot;grandson&quot;, &quot;mother&quot;, &quot;niece&quot;, &quot;sister&quot;? Possible labels: &quot;aunt&quot;, &quot;brother&quot;, &quot;daughter&quot;, &quot;daughter-in-law&quot;, &quot;father&quot;, &quot;father-in-law&quot;, &quot;granddaughter&quot;, &quot;grandfather&quot;, &quot;grandmother&quot;, &quot;grandson&quot;, &quot;mother&quot;, &quot;mother-in-law&quot;, &quot;nephew&quot;, &quot;niece&quot;, &quot;sister&quot;, &quot;son&quot;, &quot;son-in-law&quot;, &quot;uncle&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · clutrr-effc7ced7a:train:pack-f59e25835dde:exists-5

**State**

<pre>Item A:
text_A: Pedro made a pizza for his daughter. Her name is Rebecca. . One day Antonio the uncle of Rebecca, decided to surprise Rebecca with a camping trip. Rebecca had been wanting to go camping for a long time. Pedro took his daughter, Rebecca, to the father daughter dance at church. Antonio enjoys talking to his sister. Her name is Beverly.
text_B: How is Beverly related to Rebecca?

Item B:
text_A: Harold is taking his son Louis on a camping trip for the weekend. Harold loves his daughter Beverly. Louis is the great brother of Antonio. Antonio read a bedtime story to his daughter Emma.
text_B: How is Beverly related to Emma?

Item C:
text_A: Shantel made sandwiches with her daughters Dorothy and Marie. Marie &#x27;daughter Michelle made a salad. Beverly and her mother Shantel went to volunteer at the homeless shelter last weekend.
text_B: How is Beverly related to Michelle?</pre>

**Question**

<pre>Does at least one item have the label &quot;father-in-law&quot;? Possible labels: &quot;aunt&quot;, &quot;brother&quot;, &quot;daughter&quot;, &quot;daughter-in-law&quot;, &quot;father&quot;, &quot;father-in-law&quot;, &quot;granddaughter&quot;, &quot;grandfather&quot;, &quot;grandmother&quot;, &quot;grandson&quot;, &quot;mother&quot;, &quot;mother-in-law&quot;, &quot;nephew&quot;, &quot;niece&quot;, &quot;sister&quot;, &quot;son&quot;, &quot;son-in-law&quot;, &quot;uncle&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
