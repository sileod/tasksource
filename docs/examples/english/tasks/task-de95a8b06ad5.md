# disrpt/eng.dep.scidtb.rels

Classification · `multilingual-discourse-hub/disrpt`

<details><summary>Example 1</summary>

published Jev · train · disrpt-eng-dep-scidtb-rels-615591f14c:train:47

**State**

<pre>text_A: and the only manual annotations &lt;*&gt; are grammatical error labels .
text_B: needed for training</pre>

**Question**

<pre>Which discourse relation links unit text_B to unit text_A?</pre>

**Criteria (choice)**

<pre>1. attribution
2. bg-compare
3. bg-general
4. bg-goal
5. cause
6. comparison
7. condition
8. contrast
9. elab-addition
10. elab-aspect
11. elab-definition
12. elab-enumember
13. elab-example
14. elab-process_step
15. enablement
16. evaluation
17. exp-evidence
18. exp-reason
19. joint
20. manner-means
21. progression
22. result
23. summary
24. temporal</pre>

**Gold target (weight)**

<pre>9. elab-addition — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · disrpt-eng-dep-scidtb-rels-615591f14c:train:47:choice-paired-text-format

**State**

<pre>First text:
and the only manual annotations &lt;*&gt; are grammatical error labels .

Second text:
needed for training</pre>

**Question**

<pre>Which discourse relation links unit text_B to unit text_A?</pre>

**Criteria (choice)**

<pre>1. attribution
2. bg-compare
3. bg-general
4. bg-goal
5. cause
6. comparison
7. condition
8. contrast
9. elab-addition
10. elab-aspect
11. elab-definition
12. elab-enumember
13. elab-example
14. elab-process_step
15. enablement
16. evaluation
17. exp-evidence
18. exp-reason
19. joint
20. manner-means
21. progression
22. result
23. summary
24. temporal</pre>

**Gold target (weight)**

<pre>9. elab-addition — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · disrpt-eng-dep-scidtb-rels-615591f14c:train:3450

**State**

<pre>text_A: which is a common practice
text_B: to reduce computational costs , from a theoretical standpoint .</pre>

**Question**

<pre>Which discourse relation links unit text_B to unit text_A?</pre>

**Criteria (choice)**

<pre>1. attribution
2. bg-compare
3. bg-general
4. bg-goal
5. cause
6. comparison
7. condition
8. contrast
9. elab-addition
10. elab-aspect
11. elab-definition
12. elab-enumember
13. elab-example
14. elab-process_step
15. enablement
16. evaluation
17. exp-evidence
18. exp-reason
19. joint
20. manner-means
21. progression
22. result
23. summary
24. temporal</pre>

**Gold target (weight)**

<pre>15. enablement — 1</pre>

</details>
