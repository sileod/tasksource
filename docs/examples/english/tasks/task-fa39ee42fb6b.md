# nli4ct_semeval2024

Classification · `AshtonIsNotHere/nli4ct_semeval2024`

<details><summary>Example 1</summary>

published Jev · train · nli4ct-semeval2024-4a16101db5:train:0

**State**

<pre>Passage A:
INTERVENTION 1:&#32;
  Diagnostic (FLT PET)
  Patients with early stage, ER positive primary breast cancer undergo FLT PET scan at baseline and 1-6 weeks after the start of standard endocrine treatment. The surgery follows 1-7 days after the second FLT PET scan.
  Tracer used in the FLT PET (positron emission tomography) scanning procedure: [F18] fluorothymidine.
  Positron Emission Tomography: Undergo FLT PET
  Laboratory Biomarker Analysis: Correlative studies - Ki67 staining of the tumor tissue in the biopsy and surgical specimen.

Passage B:
All the primary trial participants do not receive any oral capecitabine, oral lapatinib ditosylate or cixutumumab IV, in conrast all the secondary trial subjects receive these.</pre>

**Question**

<pre>Decide whether the first text entails or contradicts the second.</pre>

**Criteria (choice)**

<pre>1. Contradiction
2. Entailment</pre>

**Gold target (weight)**

<pre>1. Contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · nli4ct-semeval2024-4a16101db5:train:0:choice-paired-text-format

**State**

<pre>Passage A:
INTERVENTION 1:&#32;
  Diagnostic (FLT PET)
  Patients with early stage, ER positive primary breast cancer undergo FLT PET scan at baseline and 1-6 weeks after the start of standard endocrine treatment. The surgery follows 1-7 days after the second FLT PET scan.
  Tracer used in the FLT PET (positron emission tomography) scanning procedure: [F18] fluorothymidine.
  Positron Emission Tomography: Undergo FLT PET
  Laboratory Biomarker Analysis: Correlative studies - Ki67 staining of the tumor tissue in the biopsy and surgical specimen.

Passage B:
All the primary trial participants do not receive any oral capecitabine, oral lapatinib ditosylate or cixutumumab IV, in conrast all the secondary trial subjects receive these.</pre>

**Question**

<pre>Decide whether the first text entails or contradicts the second.</pre>

**Criteria (choice)**

<pre>1. Contradiction
2. Entailment</pre>

**Gold target (weight)**

<pre>1. Contradiction — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · nli4ct-semeval2024-4a16101db5:train:1688

**State**

<pre>A: DISEASE CHARACTERISTICS:
  Histologically confirmed breast cancer
  Metastatic disease amenable to biopsy
  Unresected tumor with no intention to undergo resection during study
  Archival tissue from the primary diagnosis or fresh biopsy from metastatic cancer site required
  Measurable or non-measurable disease for phase I study (The Phase I portion of this study closed and the Phase II portion of the study opened as per NCCTG Addendum 6, effective January 23, 2012.)
  Measurable disease only for phase II study
  Available tumor estrogen (ER), progesterone (PR), and HER2 status from metastatic site tested by IHC or FISH OR results from the original tumor diagnosis
  Any ER, PR, or HER2 level (positive or negative) acceptable (phase I)
  Triple-negative disease only (phase II)
  ER and PR negative defined as  1% by IHC
  HER2 negative
  Patients with triple-negative breast cancer allowed provided there is clinical or radiographic evidence of tumor progression in the adjuvant or metastatic setting
  No patients whose disease can be treated with known standard therapy that is potentially curative or definitely capable of extending life expectancy
  No known CNS metastasis
  Hormone-receptor status:
  ER and PR positive or negative (phase I)
  ER and PR negative (phase II)
  PATIENT CHARACTERISTICS:
  ECOG performance status 0-1 (phase I) or 0-2 (phase II)
  Postmenopausal defined by 1 of the following:
   60 years of age
   45 years of age with last menstrual period  12 months prior and estradiol and follicle-stimulating hormone levels in postmenopausal range
  Bilateral oophorectomy
  Life expectancy  12 weeks
  ANC  1,500/mm^3
  Platelet count  100,000/mm^3
  Total bilirubin normal
  ALT and AST  3 times upper limit of normal (ULN) ( 5 times ULN if due to liver metastasis)
  Serum creatinine  1.5 times ULN
  TSH normal (thyroid hormone supplements allowed for patients with hypothyroidism)
  Not pregnant or nursing
  Fertile patients must use effective contraception
  Willing to return to Mayo Clinic or NCCTG institution (phase II) for follow-up
  Willing to provide blood samples for correlative research purposes
  No uncontrolled or intercurrent illness including, but not limited to, any of the following:
  Ongoing or active infection
  Symptomatic congestive heart failure
  Unstable angina pectoris
  Cardiac arrhythmia
  Psychiatric illness and/or social situations that would limit compliance with study requirements
  No NYHA class III or IV cardiovascular disease
  No known seizure disorder
  No co-morbid systemic illnesses or other severe concurrent disease that, in the judgment of the investigator, would make the patient inappropriate for entry into this study or interfere significantly with the proper assessment of safety and toxicity of the prescribed regimens
  No immunocompromised patients, including patients known to be HIV positive
  Immunocompromised patients due to the use of corticosteroids allowed
  No malignancy within the past 5 years except for nonmelanoma skin cancer or carcinoma in situ of the cervix
  No history of myocardial infarction  6 months
  No congenital long QT syndrome or QTcF&gt;450 msec, including:
  Complete left bundle block or use of a permanent cardiac pacemaker, history or presence of ventricular tachyarrhythmias, clinically significant resting bradycardia (&lt;50 beats per minute)
  Right bundle branch block + left anterior hemiblock (bifascicular block)
  No congestive heart failure requiring use of maintenance therapy for life-threatening ventricular arrhythmias
  PRIOR CONCURRENT THERAPY:
  See Disease Characteristics
  More than 4 weeks since prior chemotherapy or radiotherapy and fully recovered
  No radiotherapy to &gt; 25 % of bone marrow
  Prior treatments allowed (phase II):
  0 or 1 prior chemotherapy regimens for breast cancer
   2 prior aromatase-inhibitor regimens (including letrozole)
  Not currently receiving treatment in a different clinical study in which investigational procedures are performed or investigational therapies are administered
  No other concurrent investigational agent for the primary neoplasm
  No concurrent CYP3A4 inhibitors or inducers
B: Patients with non-measurable diseases are only eligible for phase I of the primary trial.</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. Contradiction
2. Entailment</pre>

**Gold target (weight)**

<pre>2. Entailment — 1</pre>

</details>
