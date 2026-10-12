# emo/emo2019

Classification · `oneonlee/cleansed_emocontext`

<details><summary>Example 1</summary>

published Jev · train · emo-emo2019-286304f33c:train:pack-62856887e4ec:label-C

**State**

<pre>Item A:
have a relationship 1 . love 2 . trust 3 . respect for each other you will become crazy

Item B:
anyway how are you im good , you ? not so good

Item C:
i am happy i can haz sum happies ? my frienda</pre>

**Question**

<pre>Choose the criterion that best describes Item C.</pre>

**Criteria (choice)**

<pre>1. angry
2. happy
3. others
4. sad</pre>

**Gold target (weight)**

<pre>2. happy — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · emo-emo2019-286304f33c:train:pack-62856887e4ec:in-B-0.2.3

**State**

<pre>Item A:
have a relationship 1 . love 2 . trust 3 . respect for each other you will become crazy

Item B:
anyway how are you im good , you ? not so good

Item C:
i am happy i can haz sum happies ? my frienda</pre>

**Question**

<pre>Is the label of Item B one of &quot;angry&quot;, &quot;others&quot;, &quot;sad&quot;? Possible labels: &quot;angry&quot;, &quot;happy&quot;, &quot;others&quot;, &quot;sad&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · emo-emo2019-286304f33c:train:pack-62856887e4ec:exists-3

**State**

<pre>Item A:
have a relationship 1 . love 2 . trust 3 . respect for each other you will become crazy

Item B:
anyway how are you im good , you ? not so good

Item C:
i am happy i can haz sum happies ? my frienda</pre>

**Question**

<pre>Does at least one item have the label &quot;sad&quot;? Possible labels: &quot;angry&quot;, &quot;happy&quot;, &quot;others&quot;, &quot;sad&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
