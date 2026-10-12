# ddi/relation

Classification · `OpenMed/DDI-Corpus-Processed`

<details><summary>Example 1</summary>

published Jev · train · ddi:train:DDI-DrugBank.d171.s0.p1:ddi-relation-568f9578b9

**State**

<pre>text_A: Tagamet, apparently through an effect on certain microsomal enzyme systems, has been reported to reduce the hepatic metabolism of warfarin-type anticoagulants, phenytoin, propranolol, nifedipine, chlordiazepoxide, diazepam, certain tricyclic antidepressants, lidocaine, theophylline and metronidazole, thereby delaying elimination and increasing blood levels of these drugs.
text_B: Drug pair: Tagamet / phenytoin</pre>

**Question**

<pre>What interaction between the two specified drugs is described in text_A?</pre>

**Criteria (choice)**

<pre>1. mechanism
2. effect
3. advice
4. interaction
5. no interaction</pre>

**Gold target (weight)**

<pre>1. mechanism — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · ddi:train:DDI-DrugBank.d466.s20.p1:ddi-relation-568f9578b9

**State**

<pre>text_A: ALLEGRA should not be taken closely in time with aluminum and magnesium containing antacids.
text_B: Drug pair: ALLEGRA / magnesium</pre>

**Question**

<pre>What interaction between the two specified drugs is described in text_A?</pre>

**Criteria (choice)**

<pre>1. mechanism
2. effect
3. advice
4. interaction
5. no interaction</pre>

**Gold target (weight)**

<pre>3. advice — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · ddi:train:DDI-DrugBank.d94.s11.p714:ddi-relation-568f9578b9

**State**

<pre>text_A: Agents that have been found, or are expected to have decreased plasma levels in the presence of EQUETROTM due to induction of CYP enzymes are the following: Acetaminophen, alprazolam, amitriptyline, bupropion, buspirone, citalopram, clobazam, clonazepam, clozapine, cyclosporin, delavirdine, desipramine, diazepam, dicumarol, doxycycline, ethosuximide, felbamate, felodipine, glucocorticoids, haloperidol, itraconazole, lamotrigine, levothyroxine, lorazepam, methadone, midazolam, mirtazapine, nortriptyline, olanzapine, oral contraceptives(3), oxcarbazepine, Phenytoin(4), praziquantel, protease inhibitors, quetiapine, risperidone, theophylline, topiramate, tiagabine, tramadol, triazolam, valproate, warfarin(5) , ziprasidone, and zonisamide.
text_B: Drug pair: haloperidol / methadone</pre>

**Question**

<pre>What interaction between the two specified drugs is described in text_A?</pre>

**Criteria (choice)**

<pre>1. mechanism
2. effect
3. advice
4. interaction
5. no interaction</pre>

**Gold target (weight)**

<pre>5. no interaction — 1</pre>

</details>
