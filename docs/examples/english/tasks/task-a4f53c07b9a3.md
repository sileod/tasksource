# silicone/oasis

Classification · `tasksource/silicone`

<details><summary>Example 1</summary>

published Jev · train · silicone-oasis-07a9efc9b7:train:pack-dda59518c41c:label-B

**State**

<pre>Item A:
the outgoing calls barred facility

Item B:
um low calls uh</pre>

**Question**

<pre>Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. accept
2. ackn
3. answ
4. answElab
5. appreciate
6. backch
7. bye
8. complete
9. confirm
10. correct
11. direct
12. directElab
13. echo
14. exclaim
15. expressOpinion
16. expressPossibility
17. expressRegret
18. expressWish
19. greet
20. hold
21. identifySelf
22. inform
23. informCont
24. informDisc
25. informIntent
26. init
27. negate
28. offer
29. pardon
30. raiseIssue
31. refer
32. refuse
33. reqDirect
34. reqInfo
35. reqModal
36. selfTalk
37. suggest
38. thank
39. informIntent-hold
40. correctSelf
41. expressRegret-inform
42. thank-identifySelf</pre>

**Gold target (weight)**

<pre>40. correctSelf — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · silicone-oasis-07a9efc9b7:train:pack-dda59518c41c:in-A-0.1.4.5.6.7.8.9.12.13.14.15.16.17.18.19.20.21.23.24.25.26.27.29.30.31.32.33.35.37.39.40.41

**State**

<pre>Item A:
the outgoing calls barred facility

Item B:
um low calls uh</pre>

**Question**

<pre>Is the label of Item A one of &quot;accept&quot;, &quot;ackn&quot;, &quot;appreciate&quot;, &quot;backch&quot;, &quot;bye&quot;, &quot;complete&quot;, &quot;confirm&quot;, &quot;correct&quot;, &quot;echo&quot;, &quot;exclaim&quot;, &quot;expressOpinion&quot;, &quot;expressPossibility&quot;, &quot;expressRegret&quot;, &quot;expressWish&quot;, &quot;greet&quot;, &quot;hold&quot;, &quot;identifySelf&quot;, &quot;inform&quot;, &quot;informDisc&quot;, &quot;informIntent&quot;, &quot;init&quot;, &quot;negate&quot;, &quot;offer&quot;, &quot;raiseIssue&quot;, &quot;refer&quot;, &quot;refuse&quot;, &quot;reqDirect&quot;, &quot;reqInfo&quot;, &quot;selfTalk&quot;, &quot;thank&quot;, &quot;correctSelf&quot;, &quot;expressRegret-inform&quot;, &quot;thank-identifySelf&quot;? Possible labels: &quot;accept&quot;, &quot;ackn&quot;, &quot;answ&quot;, &quot;answElab&quot;, &quot;appreciate&quot;, &quot;backch&quot;, &quot;bye&quot;, &quot;complete&quot;, &quot;confirm&quot;, &quot;correct&quot;, &quot;direct&quot;, &quot;directElab&quot;, &quot;echo&quot;, &quot;exclaim&quot;, &quot;expressOpinion&quot;, &quot;expressPossibility&quot;, &quot;expressRegret&quot;, &quot;expressWish&quot;, &quot;greet&quot;, &quot;hold&quot;, &quot;identifySelf&quot;, &quot;inform&quot;, &quot;informCont&quot;, &quot;informDisc&quot;, &quot;informIntent&quot;, &quot;init&quot;, &quot;negate&quot;, &quot;offer&quot;, &quot;pardon&quot;, &quot;raiseIssue&quot;, &quot;refer&quot;, &quot;refuse&quot;, &quot;reqDirect&quot;, &quot;reqInfo&quot;, &quot;reqModal&quot;, &quot;selfTalk&quot;, &quot;suggest&quot;, &quot;thank&quot;, &quot;informIntent-hold&quot;, &quot;correctSelf&quot;, &quot;expressRegret-inform&quot;, &quot;thank-identifySelf&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · silicone-oasis-07a9efc9b7:train:pack-dda59518c41c:exists-29

**State**

<pre>Item A:
the outgoing calls barred facility

Item B:
um low calls uh</pre>

**Question**

<pre>Does at least one item have the label &quot;raiseIssue&quot;? Possible labels: &quot;accept&quot;, &quot;ackn&quot;, &quot;answ&quot;, &quot;answElab&quot;, &quot;appreciate&quot;, &quot;backch&quot;, &quot;bye&quot;, &quot;complete&quot;, &quot;confirm&quot;, &quot;correct&quot;, &quot;direct&quot;, &quot;directElab&quot;, &quot;echo&quot;, &quot;exclaim&quot;, &quot;expressOpinion&quot;, &quot;expressPossibility&quot;, &quot;expressRegret&quot;, &quot;expressWish&quot;, &quot;greet&quot;, &quot;hold&quot;, &quot;identifySelf&quot;, &quot;inform&quot;, &quot;informCont&quot;, &quot;informDisc&quot;, &quot;informIntent&quot;, &quot;init&quot;, &quot;negate&quot;, &quot;offer&quot;, &quot;pardon&quot;, &quot;raiseIssue&quot;, &quot;refer&quot;, &quot;refuse&quot;, &quot;reqDirect&quot;, &quot;reqInfo&quot;, &quot;reqModal&quot;, &quot;selfTalk&quot;, &quot;suggest&quot;, &quot;thank&quot;, &quot;informIntent-hold&quot;, &quot;correctSelf&quot;, &quot;expressRegret-inform&quot;, &quot;thank-identifySelf&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
