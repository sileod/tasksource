# trec

Classification · `tasksource/trec`

<details><summary>Example 1</summary>

published Jev · train · trec-8dbd212e0d:train:3007

**State**

<pre>What state capital comes last alphabetically ?</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. ABBR:abb
2. ABBR:exp
3. ENTY:animal
4. ENTY:body
5. ENTY:color
6. ENTY:cremat
7. ENTY:currency
8. ENTY:dismed
9. ENTY:event
10. ENTY:food
11. ENTY:instru
12. ENTY:lang
13. ENTY:letter
14. ENTY:other
15. ENTY:plant
16. ENTY:product
17. ENTY:religion
18. ENTY:sport
19. ENTY:substance
20. ENTY:symbol
21. ENTY:techmeth
22. ENTY:termeq
23. ENTY:veh
24. ENTY:word
25. DESC:def
26. DESC:desc
27. DESC:manner
28. DESC:reason
29. HUM:gr
30. HUM:ind
31. HUM:title
32. HUM:desc
33. LOC:city
34. LOC:country
35. LOC:mount
36. LOC:other
37. LOC:state
38. NUM:code
39. NUM:count
40. NUM:date
41. NUM:dist
42. NUM:money
43. NUM:ord
44. NUM:other
45. NUM:period
46. NUM:perc
47. NUM:speed
48. NUM:temp
49. NUM:volsize
50. NUM:weight</pre>

**Gold target (weight)**

<pre>33. LOC:city — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · trec-8dbd212e0d:train:2272

**State**

<pre>Where can I find a review of Nightmare on Elm Street in a film journal ?</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. ABBR:abb
2. ABBR:exp
3. ENTY:animal
4. ENTY:body
5. ENTY:color
6. ENTY:cremat
7. ENTY:currency
8. ENTY:dismed
9. ENTY:event
10. ENTY:food
11. ENTY:instru
12. ENTY:lang
13. ENTY:letter
14. ENTY:other
15. ENTY:plant
16. ENTY:product
17. ENTY:religion
18. ENTY:sport
19. ENTY:substance
20. ENTY:symbol
21. ENTY:techmeth
22. ENTY:termeq
23. ENTY:veh
24. ENTY:word
25. DESC:def
26. DESC:desc
27. DESC:manner
28. DESC:reason
29. HUM:gr
30. HUM:ind
31. HUM:title
32. HUM:desc
33. LOC:city
34. LOC:country
35. LOC:mount
36. LOC:other
37. LOC:state
38. NUM:code
39. NUM:count
40. NUM:date
41. NUM:dist
42. NUM:money
43. NUM:ord
44. NUM:other
45. NUM:period
46. NUM:perc
47. NUM:speed
48. NUM:temp
49. NUM:volsize
50. NUM:weight</pre>

**Gold target (weight)**

<pre>36. LOC:other — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · trec-8dbd212e0d:train:2272:noul-label-verification

**State**

<pre>Where can I find a review of Nightmare on Elm Street in a film journal ?</pre>

**Question**

<pre>Is &quot;LOC:other&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
