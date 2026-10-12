# robustLR

Classification · `tasksource/robustLR`

<details><summary>Example 1</summary>

published Jev · train · robustLR-408a7e4eae:train:pack-cb0a463f4bf7:label-A

**State**

<pre>Item A:
text_A: Anne is not green. Harry is the uncle of Erin. Harry is nice if Anne is not green. If The uncle of Erin is Harry then Erin is young. Bob is Dave&#x27;s mother if Harry is nice. If Bob is Dave&#x27;s mother then Harry is not the brother of Anne. The brother of Anne is Harry if The grandmother of Fiona is not Erin. If Erin is not Fiona&#x27;s grandmother then Erin is cold. If Harry is not Anne&#x27;s brother then Charlie is Gary&#x27;s sister. If Erin is not Fiona&#x27;s grandmother then The sister of Gary is not Charlie. The grandmother of Charlie is not Gary if The brother of Anne is not Harry. Gary is Charlie&#x27;s grandmother if Erin is cold. If Gary is not Charlie&#x27;s grandmother then Erin is green. If Erin is not Fiona&#x27;s grandmother then Erin is not green.
text_B: Erin is green.

Item B:
text_A: Harry is Gary&#x27;s brother. The brother of Bob is not Harry. If Harry is Gary&#x27;s brother then Dave is not big. If Erin is not young then Dave is big. Harry is young if Harry is the brother of Gary. Harry is not young if Erin is not green. Gary is smart if The brother of Gary is Harry. If Erin is not young then Gary is not smart. If Dave is not big then Fiona is not the mother of Erin. The mother of Erin is Fiona if Gary is not Harry&#x27;s mother. Bob is not blue if Erin is not green. Charlie is round if Dave is not big. If Erin is not green then Charlie is not round. Anne is not Gary&#x27;s son if Gary is smart. If Erin is not green then Anne is the son of Gary. If Charlie is round then The grandmother of Anne is not Gary. Gary is the grandmother of Anne if The mother of Harry is not Gary. The sister of Fiona is not Dave if Charlie is round. The sister of Fiona is Dave if Erin is not young. If Harry is Gary&#x27;s brother then Gary is cold. If Erin is not young then Gary is not cold. If Gary is not the mother of Harry then Gary is quiet. Fiona is not the daughter of Charlie if Erin is not young. If The grandmother of Anne is not Gary then Harry is not smart. Harry is smart if Gary is quiet. If Dave is not Fiona&#x27;s sister then Dave is cold. Dave is not cold if The mother of Harry is not Gary. If Dave is cold then Bob is not white. Bob is white if Erin is not young. If The brother of Gary is Harry then Gary is green. If Erin is not young then Gary is not green. The husband of Bob is not Harry if Erin is not green.
text_B: Dave is not cold.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · robustLR-408a7e4eae:train:pack-cb0a463f4bf7:in-B-1.2

**State**

<pre>Item A:
text_A: Anne is not green. Harry is the uncle of Erin. Harry is nice if Anne is not green. If The uncle of Erin is Harry then Erin is young. Bob is Dave&#x27;s mother if Harry is nice. If Bob is Dave&#x27;s mother then Harry is not the brother of Anne. The brother of Anne is Harry if The grandmother of Fiona is not Erin. If Erin is not Fiona&#x27;s grandmother then Erin is cold. If Harry is not Anne&#x27;s brother then Charlie is Gary&#x27;s sister. If Erin is not Fiona&#x27;s grandmother then The sister of Gary is not Charlie. The grandmother of Charlie is not Gary if The brother of Anne is not Harry. Gary is Charlie&#x27;s grandmother if Erin is cold. If Gary is not Charlie&#x27;s grandmother then Erin is green. If Erin is not Fiona&#x27;s grandmother then Erin is not green.
text_B: Erin is green.

Item B:
text_A: Harry is Gary&#x27;s brother. The brother of Bob is not Harry. If Harry is Gary&#x27;s brother then Dave is not big. If Erin is not young then Dave is big. Harry is young if Harry is the brother of Gary. Harry is not young if Erin is not green. Gary is smart if The brother of Gary is Harry. If Erin is not young then Gary is not smart. If Dave is not big then Fiona is not the mother of Erin. The mother of Erin is Fiona if Gary is not Harry&#x27;s mother. Bob is not blue if Erin is not green. Charlie is round if Dave is not big. If Erin is not green then Charlie is not round. Anne is not Gary&#x27;s son if Gary is smart. If Erin is not green then Anne is the son of Gary. If Charlie is round then The grandmother of Anne is not Gary. Gary is the grandmother of Anne if The mother of Harry is not Gary. The sister of Fiona is not Dave if Charlie is round. The sister of Fiona is Dave if Erin is not young. If Harry is Gary&#x27;s brother then Gary is cold. If Erin is not young then Gary is not cold. If Gary is not the mother of Harry then Gary is quiet. Fiona is not the daughter of Charlie if Erin is not young. If The grandmother of Anne is not Gary then Harry is not smart. Harry is smart if Gary is quiet. If Dave is not Fiona&#x27;s sister then Dave is cold. Dave is not cold if The mother of Harry is not Gary. If Dave is cold then Bob is not white. Bob is white if Erin is not young. If The brother of Gary is Harry then Gary is green. If Erin is not young then Gary is not green. The husband of Bob is not Harry if Erin is not green.
text_B: Dave is not cold.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Is the label of Item B one of &quot;neutral&quot;, &quot;contradiction&quot;? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · robustLR-408a7e4eae:train:pack-cb0a463f4bf7:all-same

**State**

<pre>Item A:
text_A: Anne is not green. Harry is the uncle of Erin. Harry is nice if Anne is not green. If The uncle of Erin is Harry then Erin is young. Bob is Dave&#x27;s mother if Harry is nice. If Bob is Dave&#x27;s mother then Harry is not the brother of Anne. The brother of Anne is Harry if The grandmother of Fiona is not Erin. If Erin is not Fiona&#x27;s grandmother then Erin is cold. If Harry is not Anne&#x27;s brother then Charlie is Gary&#x27;s sister. If Erin is not Fiona&#x27;s grandmother then The sister of Gary is not Charlie. The grandmother of Charlie is not Gary if The brother of Anne is not Harry. Gary is Charlie&#x27;s grandmother if Erin is cold. If Gary is not Charlie&#x27;s grandmother then Erin is green. If Erin is not Fiona&#x27;s grandmother then Erin is not green.
text_B: Erin is green.

Item B:
text_A: Harry is Gary&#x27;s brother. The brother of Bob is not Harry. If Harry is Gary&#x27;s brother then Dave is not big. If Erin is not young then Dave is big. Harry is young if Harry is the brother of Gary. Harry is not young if Erin is not green. Gary is smart if The brother of Gary is Harry. If Erin is not young then Gary is not smart. If Dave is not big then Fiona is not the mother of Erin. The mother of Erin is Fiona if Gary is not Harry&#x27;s mother. Bob is not blue if Erin is not green. Charlie is round if Dave is not big. If Erin is not green then Charlie is not round. Anne is not Gary&#x27;s son if Gary is smart. If Erin is not green then Anne is the son of Gary. If Charlie is round then The grandmother of Anne is not Gary. Gary is the grandmother of Anne if The mother of Harry is not Gary. The sister of Fiona is not Dave if Charlie is round. The sister of Fiona is Dave if Erin is not young. If Harry is Gary&#x27;s brother then Gary is cold. If Erin is not young then Gary is not cold. If Gary is not the mother of Harry then Gary is quiet. Fiona is not the daughter of Charlie if Erin is not young. If The grandmother of Anne is not Gary then Harry is not smart. Harry is smart if Gary is quiet. If Dave is not Fiona&#x27;s sister then Dave is cold. Dave is not cold if The mother of Harry is not Gary. If Dave is cold then Bob is not white. Bob is white if Erin is not young. If The brother of Gary is Harry then Gary is green. If Erin is not young then Gary is not green. The husband of Bob is not Harry if Erin is not green.
text_B: Dave is not cold.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B, contradict it, or neither?&quot;
Do all items have the same label? Possible labels: &quot;entailment&quot;, &quot;neutral&quot;, &quot;contradiction&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>
