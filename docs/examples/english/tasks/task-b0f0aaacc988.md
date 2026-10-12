# defeasible-nli/social

Classification · `tasksource/defeasible-nli`

<details><summary>Example 1</summary>

published Jev · train · defeasible-nli-social-7479767344:train:pack-3b97ea6e5757:label-A

**State**

<pre>Item A:
text_A: It&#x27;s wrong to do something to sway someone&#x27;s decisions about things when it&#x27;s not what&#x27;s best for them.
text_B: it&#x27;s the best for a large number of people

Item B:
text_A: It&#x27;s okay to want someone you care about to go out of their way and do something nice for you.
text_B: You did the same for them and want it in return.

Item C:
text_A: You have right to ask neighbors not to disturb your peace and quiet.
text_B: They are in serious need of help.

Item D:
text_A: You should be considerate of your neighbors.
text_B: they have treated you very well.</pre>

**Question**

<pre>Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. strengthener
2. weakener</pre>

**Gold target (weight)**

<pre>2. weakener — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · defeasible-nli-social-7479767344:train:pack-3b97ea6e5757:in-B-0

**State**

<pre>Item A:
text_A: It&#x27;s wrong to do something to sway someone&#x27;s decisions about things when it&#x27;s not what&#x27;s best for them.
text_B: it&#x27;s the best for a large number of people

Item B:
text_A: It&#x27;s okay to want someone you care about to go out of their way and do something nice for you.
text_B: You did the same for them and want it in return.

Item C:
text_A: You have right to ask neighbors not to disturb your peace and quiet.
text_B: They are in serious need of help.

Item D:
text_A: You should be considerate of your neighbors.
text_B: they have treated you very well.</pre>

**Question**

<pre>Is the label of Item B &quot;strengthener&quot;? Possible labels: &quot;strengthener&quot;, &quot;weakener&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · defeasible-nli-social-7479767344:train:pack-3b97ea6e5757:exists-0

**State**

<pre>Item A:
text_A: It&#x27;s wrong to do something to sway someone&#x27;s decisions about things when it&#x27;s not what&#x27;s best for them.
text_B: it&#x27;s the best for a large number of people

Item B:
text_A: It&#x27;s okay to want someone you care about to go out of their way and do something nice for you.
text_B: You did the same for them and want it in return.

Item C:
text_A: You have right to ask neighbors not to disturb your peace and quiet.
text_B: They are in serious need of help.

Item D:
text_A: You should be considerate of your neighbors.
text_B: they have treated you very well.</pre>

**Question**

<pre>Does at least one item have the label &quot;strengthener&quot;? Possible labels: &quot;strengthener&quot;, &quot;weakener&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
