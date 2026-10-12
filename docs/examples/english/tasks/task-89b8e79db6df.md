# Pol_NLI

Classification · `mlburnham/Pol_NLI`

<details><summary>Example 1</summary>

published Jev · train · Pol-NLI-e6c0484ea5:train:pack-00c68a1beec2:label-A

**State**

<pre>Item A:
text_A: A city maintained a rotation list of companies that provided towing services at the request of the city&#x27;s police department. Departing from a prior policy of removing a tow truck operator from the list only for cause, the city removed an operator from the list after the operator refused to contribute to the re-election campaign of the city&#x27;s mayor and instead supported the mayor&#x27;s opponent and displayed the opponent&#x27;s campaign posters at the operator&#x27;s place of business. The operator had been on the list since 1965. The operator brought an action against the city in the United States District Court for the Northern District of Illinois, in which it alleged that its removal from the list (1) was in retaliation for the operator&#x27;s stance in the mayoral campaign, (2) caused it to lose substantial income, and (3) infringed upon its rights under the Federal Constitution&#x27;s First Amendment, in violation of 42 USCS 1983. The District Court, in dismissing the complaint, held, in conformity with binding Seventh Circuit precedent, that the protections afforded by the United States Supreme Court decisions in Elrod v Burns (1976) 427 US 347, 49 L Ed 2d 547, 96 S Ct 2673, and Branti v Finkel (1980) 445 US 507, 63 L Ed 2d 574, 100 S Ct 1287--holding that government officials may not discharge public employees for refusing to support a political party or its candidates, unless political affiliation is a reasonably appropriate requirement for the job in question--did not extend to independent contractors (843 F Supp 1231). The United States Court of Appeals for the Seventh Circuit, in affirming, adhered to the view that it should be up to the Supreme Court to extend Elrod (47 F3d 883).
text_B: This court case is about first amendment.

Item B:
text_A: Long-Term Unemployment Elimination Act of 2021
This bill creates a federal program to generate job opportunities for full-time students and certain workers who have been unemployed for at least 27 weeks.
Specifically, the Department of Labor must establish and administer a jobs initiative for eligible workers (i.e., individuals who are at least 18 years of age, are authorized to be employed in the United States, have not been employed or are full-time students for at least 27 weeks, and are currently seeking or have been seeking employment for at least 4 weeks), carried out through local boards or community-based organizations.
Labor shall award grants to local boards and community-based organizations to carry out jobs programs, including any added costs related to them in order to achieve economic development and job growth. Grant recipients must serve (1) high-poverty areas; or (2) areas for which the prime working-age employment-to-population ratio has been low, relative to that ratio for the United States for at least three years.
text_B: This text is about labor and employment.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · Pol-NLI-e6c0484ea5:train:pack-00c68a1beec2:in-A-1

**State**

<pre>Item A:
text_A: A city maintained a rotation list of companies that provided towing services at the request of the city&#x27;s police department. Departing from a prior policy of removing a tow truck operator from the list only for cause, the city removed an operator from the list after the operator refused to contribute to the re-election campaign of the city&#x27;s mayor and instead supported the mayor&#x27;s opponent and displayed the opponent&#x27;s campaign posters at the operator&#x27;s place of business. The operator had been on the list since 1965. The operator brought an action against the city in the United States District Court for the Northern District of Illinois, in which it alleged that its removal from the list (1) was in retaliation for the operator&#x27;s stance in the mayoral campaign, (2) caused it to lose substantial income, and (3) infringed upon its rights under the Federal Constitution&#x27;s First Amendment, in violation of 42 USCS 1983. The District Court, in dismissing the complaint, held, in conformity with binding Seventh Circuit precedent, that the protections afforded by the United States Supreme Court decisions in Elrod v Burns (1976) 427 US 347, 49 L Ed 2d 547, 96 S Ct 2673, and Branti v Finkel (1980) 445 US 507, 63 L Ed 2d 574, 100 S Ct 1287--holding that government officials may not discharge public employees for refusing to support a political party or its candidates, unless political affiliation is a reasonably appropriate requirement for the job in question--did not extend to independent contractors (843 F Supp 1231). The United States Court of Appeals for the Seventh Circuit, in affirming, adhered to the view that it should be up to the Supreme Court to extend Elrod (47 F3d 883).
text_B: This court case is about first amendment.

Item B:
text_A: Long-Term Unemployment Elimination Act of 2021
This bill creates a federal program to generate job opportunities for full-time students and certain workers who have been unemployed for at least 27 weeks.
Specifically, the Department of Labor must establish and administer a jobs initiative for eligible workers (i.e., individuals who are at least 18 years of age, are authorized to be employed in the United States, have not been employed or are full-time students for at least 27 weeks, and are currently seeking or have been seeking employment for at least 4 weeks), carried out through local boards or community-based organizations.
Labor shall award grants to local boards and community-based organizations to carry out jobs programs, including any added costs related to them in order to achieve economic development and job growth. Grant recipients must serve (1) high-poverty areas; or (2) areas for which the prime working-age employment-to-population ratio has been low, relative to that ratio for the United States for at least three years.
text_B: This text is about labor and employment.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Is the label of Item A &quot;not_entailment&quot;? Possible labels: &quot;entailment&quot;, &quot;not_entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · Pol-NLI-e6c0484ea5:train:pack-00c68a1beec2:all-same

**State**

<pre>Item A:
text_A: A city maintained a rotation list of companies that provided towing services at the request of the city&#x27;s police department. Departing from a prior policy of removing a tow truck operator from the list only for cause, the city removed an operator from the list after the operator refused to contribute to the re-election campaign of the city&#x27;s mayor and instead supported the mayor&#x27;s opponent and displayed the opponent&#x27;s campaign posters at the operator&#x27;s place of business. The operator had been on the list since 1965. The operator brought an action against the city in the United States District Court for the Northern District of Illinois, in which it alleged that its removal from the list (1) was in retaliation for the operator&#x27;s stance in the mayoral campaign, (2) caused it to lose substantial income, and (3) infringed upon its rights under the Federal Constitution&#x27;s First Amendment, in violation of 42 USCS 1983. The District Court, in dismissing the complaint, held, in conformity with binding Seventh Circuit precedent, that the protections afforded by the United States Supreme Court decisions in Elrod v Burns (1976) 427 US 347, 49 L Ed 2d 547, 96 S Ct 2673, and Branti v Finkel (1980) 445 US 507, 63 L Ed 2d 574, 100 S Ct 1287--holding that government officials may not discharge public employees for refusing to support a political party or its candidates, unless political affiliation is a reasonably appropriate requirement for the job in question--did not extend to independent contractors (843 F Supp 1231). The United States Court of Appeals for the Seventh Circuit, in affirming, adhered to the view that it should be up to the Supreme Court to extend Elrod (47 F3d 883).
text_B: This court case is about first amendment.

Item B:
text_A: Long-Term Unemployment Elimination Act of 2021
This bill creates a federal program to generate job opportunities for full-time students and certain workers who have been unemployed for at least 27 weeks.
Specifically, the Department of Labor must establish and administer a jobs initiative for eligible workers (i.e., individuals who are at least 18 years of age, are authorized to be employed in the United States, have not been employed or are full-time students for at least 27 weeks, and are currently seeking or have been seeking employment for at least 4 weeks), carried out through local boards or community-based organizations.
Labor shall award grants to local boards and community-based organizations to carry out jobs programs, including any added costs related to them in order to achieve economic development and job growth. Grant recipients must serve (1) high-poverty areas; or (2) areas for which the prime working-age employment-to-population ratio has been low, relative to that ratio for the United States for at least three years.
text_B: This text is about labor and employment.</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Do all items have the same label? Possible labels: &quot;entailment&quot;, &quot;not_entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
