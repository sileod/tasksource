# pragmeval/emergent

Classification · `sileod/pragmeval`

<details><summary>Example 1</summary>

published Jev · train · pragmeval-emergent-001ac610b1:train:pack-09797364e5b3:label-B

**State**

<pre>Item A:
text_A: a florida woman got a third breast
text_B: 3-boobed woman a fake

Item B:
text_A: a wife and child of isis leader abu bakr al-baghdadi were detained in lebanon
text_B: iraq says arrested woman is not the wife of isis leader al-baghdadi

Item C:
text_A: tiger woods is serving a suspension from the pga tour after failing a drug test
text_B: pga tour denies claim that tiger woods failed a drug test

Item D:
text_A: mcdonald&#x27;s will stop serving the big mac
text_B: mcdonald&#x27;s big mac hoax: removing apple pies and eliminating large size options report isn&#x27;t real</pre>

**Question**

<pre>Each item answers: &quot;What stance does article headline text_B take toward the claim in text_A?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. observing
2. for
3. against</pre>

**Gold target (weight)**

<pre>3. against — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · pragmeval-emergent-001ac610b1:train:pack-09797364e5b3:same-B-C

**State**

<pre>Item A:
text_A: a florida woman got a third breast
text_B: 3-boobed woman a fake

Item B:
text_A: a wife and child of isis leader abu bakr al-baghdadi were detained in lebanon
text_B: iraq says arrested woman is not the wife of isis leader al-baghdadi

Item C:
text_A: tiger woods is serving a suspension from the pga tour after failing a drug test
text_B: pga tour denies claim that tiger woods failed a drug test

Item D:
text_A: mcdonald&#x27;s will stop serving the big mac
text_B: mcdonald&#x27;s big mac hoax: removing apple pies and eliminating large size options report isn&#x27;t real</pre>

**Question**

<pre>Each item answers: &quot;What stance does article headline text_B take toward the claim in text_A?&quot;
Do Item B and Item C have the same label? Possible labels: &quot;observing&quot;, &quot;for&quot;, &quot;against&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · pragmeval-emergent-001ac610b1:train:pack-09797364e5b3:all-same

**State**

<pre>Item A:
text_A: a florida woman got a third breast
text_B: 3-boobed woman a fake

Item B:
text_A: a wife and child of isis leader abu bakr al-baghdadi were detained in lebanon
text_B: iraq says arrested woman is not the wife of isis leader al-baghdadi

Item C:
text_A: tiger woods is serving a suspension from the pga tour after failing a drug test
text_B: pga tour denies claim that tiger woods failed a drug test

Item D:
text_A: mcdonald&#x27;s will stop serving the big mac
text_B: mcdonald&#x27;s big mac hoax: removing apple pies and eliminating large size options report isn&#x27;t real</pre>

**Question**

<pre>Each item answers: &quot;What stance does article headline text_B take toward the claim in text_A?&quot;
Do all items have the same label? Possible labels: &quot;observing&quot;, &quot;for&quot;, &quot;against&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
