# glue/rte

Classification · `nyu-mll/glue`

<details><summary>Example 1</summary>

published Jev · train · glue-rte-2dc856f29b:train:31

**State**

<pre>text_A: Archaeologists have found approximately 30 beautifully preserved mummies in a 4,000 year old Egyptian necropolis which held 53 tombs. Supervisor of Antiquities for Middle Egypt Dr. Abdel-Rahman El-Ayedi&#x27;s team established his archaeological site in the Faiyum Oasis near the El-Lahun Egyptian pyramid which is just south of Cairo. Besides the mummies, the team found masks, amulets, clay pots and an offering table located in a funerary chapel. The chapel dates back to about 30 BC to 337 AD.
text_B: 30 beautifully preserved mummies have been located in the south of Cairo.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · glue-rte-2dc856f29b:train:31:noul-label-verification

**State**

<pre>text_A: Archaeologists have found approximately 30 beautifully preserved mummies in a 4,000 year old Egyptian necropolis which held 53 tombs. Supervisor of Antiquities for Middle Egypt Dr. Abdel-Rahman El-Ayedi&#x27;s team established his archaeological site in the Faiyum Oasis near the El-Lahun Egyptian pyramid which is just south of Cairo. Besides the mummies, the team found masks, amulets, clay pots and an offering table located in a funerary chapel. The chapel dates back to about 30 BC to 337 AD.
text_B: 30 beautifully preserved mummies have been located in the south of Cairo.</pre>

**Question**

<pre>Does text_A entail text_B? Is &quot;not_entailment&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · glue-rte-2dc856f29b:train:1786

**State**

<pre>text_A: A member of the Burmese delegation, Thaung Tun, told reporters on the eve of the meeting that his government did not wish to place ASEAN in a difficult position by insisting on assuming the chairmanship.
text_B: Thaung Tun is a representative of Burma.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
