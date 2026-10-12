# autotnli

Classification · `tasksource/autotnli`

<details><summary>Example 1</summary>

published Jev · train · autotnli-dc994b1381:train:pack-351309f38308:label-A

**State**

<pre>Item A:
text_A: It has 82000 employees . pharmaceuticals,business-to-business are the industries this company works in . This company was found on 2010 . Global are the areas served by Actua Corporation . Products made by this company are Securities . Logosoft d.o.o. was the predecessor company . www.foxconn.com is the name of it&#x27;s website . Bellevue, Washington, United States is where the headquarters are located . Actua Corporation was founded by Ken Fox . The key people of the company is Terry Downes . The trading names of this company are TYO: 2587 . The owner of this company is Liaison Interactive (SZSE: 002280) . Actua Corporation is a parent company of Claro . Hasee Group is a parent of Actua Corporation . Aktieselskab is the type of the company
text_B: This is traded as SAZ in FWB

Item B:
text_A: Temple of Caesar is the resting place of the person . Joseph Engelberger played Trombone instruments . The person&#x27;s age was 90 . Joseph Engelberger was married with  Angelina Jolie , ( m. 2014;  sep. 2016) . Joseph Engelberger sings in A&amp;M labels . DaveGahan.com is this person&#x27;s website . The charges against this person are Indecent exposure . The doctoral students of Joseph Engelberger are Max Perutz[citation needed] . music hall these are the genres in which Joseph Engelberger sings . Engineer, singer-songwriter are the occupations of this person . May 13,1999 is when Joseph Engelberger died . 2 is the number of kids that the person has . The person worked at University of Strassburg institutions . Joseph Engelberger had a residence at Streeterville . Japan Prize (1997), Rumford Medal (1816) was won by Joseph Engelberger . July 28, 1959 is when Joseph Engelberger was born
text_B: The faculty mentored Max Perutz[citation needed]</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral</pre>

**Gold target (weight)**

<pre>2. neutral — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · autotnli-dc994b1381:train:pack-351309f38308:same-A-B

**State**

<pre>Item A:
text_A: It has 82000 employees . pharmaceuticals,business-to-business are the industries this company works in . This company was found on 2010 . Global are the areas served by Actua Corporation . Products made by this company are Securities . Logosoft d.o.o. was the predecessor company . www.foxconn.com is the name of it&#x27;s website . Bellevue, Washington, United States is where the headquarters are located . Actua Corporation was founded by Ken Fox . The key people of the company is Terry Downes . The trading names of this company are TYO: 2587 . The owner of this company is Liaison Interactive (SZSE: 002280) . Actua Corporation is a parent company of Claro . Hasee Group is a parent of Actua Corporation . Aktieselskab is the type of the company
text_B: This is traded as SAZ in FWB

Item B:
text_A: Temple of Caesar is the resting place of the person . Joseph Engelberger played Trombone instruments . The person&#x27;s age was 90 . Joseph Engelberger was married with  Angelina Jolie , ( m. 2014;  sep. 2016) . Joseph Engelberger sings in A&amp;M labels . DaveGahan.com is this person&#x27;s website . The charges against this person are Indecent exposure . The doctoral students of Joseph Engelberger are Max Perutz[citation needed] . music hall these are the genres in which Joseph Engelberger sings . Engineer, singer-songwriter are the occupations of this person . May 13,1999 is when Joseph Engelberger died . 2 is the number of kids that the person has . The person worked at University of Strassburg institutions . Joseph Engelberger had a residence at Streeterville . Japan Prize (1997), Rumford Medal (1816) was won by Joseph Engelberger . July 28, 1959 is when Joseph Engelberger was born
text_B: The faculty mentored Max Perutz[citation needed]</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Do Item A and Item B have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · autotnli-dc994b1381:train:pack-351309f38308:all-same

**State**

<pre>Item A:
text_A: It has 82000 employees . pharmaceuticals,business-to-business are the industries this company works in . This company was found on 2010 . Global are the areas served by Actua Corporation . Products made by this company are Securities . Logosoft d.o.o. was the predecessor company . www.foxconn.com is the name of it&#x27;s website . Bellevue, Washington, United States is where the headquarters are located . Actua Corporation was founded by Ken Fox . The key people of the company is Terry Downes . The trading names of this company are TYO: 2587 . The owner of this company is Liaison Interactive (SZSE: 002280) . Actua Corporation is a parent company of Claro . Hasee Group is a parent of Actua Corporation . Aktieselskab is the type of the company
text_B: This is traded as SAZ in FWB

Item B:
text_A: Temple of Caesar is the resting place of the person . Joseph Engelberger played Trombone instruments . The person&#x27;s age was 90 . Joseph Engelberger was married with  Angelina Jolie , ( m. 2014;  sep. 2016) . Joseph Engelberger sings in A&amp;M labels . DaveGahan.com is this person&#x27;s website . The charges against this person are Indecent exposure . The doctoral students of Joseph Engelberger are Max Perutz[citation needed] . music hall these are the genres in which Joseph Engelberger sings . Engineer, singer-songwriter are the occupations of this person . May 13,1999 is when Joseph Engelberger died . 2 is the number of kids that the person has . The person worked at University of Strassburg institutions . Joseph Engelberger had a residence at Streeterville . Japan Prize (1997), Rumford Medal (1816) was won by Joseph Engelberger . July 28, 1959 is when Joseph Engelberger was born
text_B: The faculty mentored Max Perutz[citation needed]</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Do all items have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
