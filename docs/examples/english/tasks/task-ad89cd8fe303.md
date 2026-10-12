# coqa/yesno

Classification · `stanfordnlp/coqa`

<details><summary>Example 1</summary>

published Jev · train · coqa:train:5a5b3edd97c4d2ca071f0c67a98cf97ff717c08b3183cf9af389503b548ae7fe:11:coqa-yesno-6b0f120687

**State**

<pre>text_A: KUPANG, Indonesia (CNN) -- At Roslin Orphanage, children giggle through deep concentration as they try to master the &quot;Chicken Dance.&quot; It&#x27;s a far cry from the Indonesian orphans&#x27; earlier months and years.&#32;

Budi Soehardi poses with young residents of Roslin Orphanage.&#32;

&quot;They are cheerful-looking and photogenic, but close to all have a very sad story,&quot; said Budi Soehardi, founder of the West Timor orphanage.&#32;

&quot;Some of the babies come because a mother passes away right after delivery because of lack of nutrition. Others come from extreme poverty. Some come from families [that] just do not want the children and abandon them,&quot; he said. Vote now for the CNN Hero of the Year&#32;

Soehardi, a 53-year-old Indonesian pilot living in Singapore, and his wife, Peggy, look after 47 children at the orphanage. They have a personal relationship with each one, and consider them part of their family. The couple named many of the children since they entered the orphanage as babies -- some of them tiny victims and refugees from the conflict in East Timor.&#32;

Soehardi has three children of his own but says there is no difference between what he supplies for his biological children and those living at the orphanage. They all get clean living spaces, vaccinations, food, clothing and vitamins from the United States.&#32;

&quot;Mr. Budi is like my own father,&quot; said Gerson Mangi, 20, a resident at Roslin Orphanage. Mangi, who came to the orphanage when he was 12 years old, had no means to attend school after his parents died. Now, thanks to the educational training at Roslin and a private sponsor, he is in medical school.&#32;
Earlier dialogue:
Question: how many kids live in the orphanage?
Answer: 47
Question: who runs it?
Answer: Soehardi
Question: first name?
Answer: Budi
Question: his age?
Answer: 53
Question: is he married?
Answer: Yes
Question: her name?
Answer: Peggy
Question: does he have his own kids?
Answer: Yes
Question: 4 of them?
Answer: No
Question: how many?
Answer: three children
Question: does he treat his own kids different?
Answer: No
Question: what does he provide for the kids?
Answer: clean living spaces
text_B: anything else?</pre>

**Question**

<pre>Answer the current question in text_B using the passage and earlier dialogue in text_A.</pre>

**Criteria (choice)**

<pre>1. no
2. yes</pre>

**Gold target (weight)**

<pre>2. yes — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · coqa:train:684b94e8ad411cff90eb36ad1143b1f0235400fe4cf38ab9ccc30eec2ebe4f26:9:coqa-yesno-6b0f120687

**State**

<pre>text_A: (CNN) -- Libya&#x27;s transitional government would prefer to try former leader Moammar Gadhafi&#x27;s son Saif al-Islam in a Libyan court despite an arrest warrant issued for him by the International Criminal Court, a spokesman said Sunday.&#32;

The court&#x27;s chief prosecutor said Friday it was having &quot;informal conversations&quot; about the surrender of Saif al-Islam Gadhafi, who is wanted for crimes against humanity. But Luis Moreno-Ocampo said the court did not know his whereabouts and would not reveal with whom the court is talking.&#32;

Saif al-Islam Gadhafi has realized there is no escape, and wants to negotiate a surrender to limit the damage, said Anees al-Sharif, spokesman for the National Transitional Council&#x27;s Tripoli Military Council.&#32;

But, he said, if the younger Gadhafi is captured in Libya, he will be tried there, as his alleged crimes were committed on Libyan soil. If he is captured outside Libya, the location of his trial will depend on where he is captured because of the ICC arrest warrants, he said.&#32;

However, the NTC would prefer to try Saif al-Islam Gadhafi in a Libyan court, as it would like to show that he could receive a fair trial, he said.&#32;

Moreno-Ocampo said Friday that if Saif al-Islam Gadhafi is brought before the ICC, he will &quot;have all the rights and be protected,&quot; and will be allowed to present a defense.&#32;

&quot;We believe we have a strong case,&quot; the prosecutor told CNN in an exclusive interview from The Hague. &quot;We believe he should be convicted.&quot;&#32;

The court believes Saif al-Islam Gadhafi, along with his father and his brother-in-law, Abdulla al-Sanussi, are responsible for crimes against humanity including murder and persecution across the country beginning in February amid anti-government demonstrations, Moreno-Ocampo said.&#32;
Earlier dialogue:
Question: Who has a warrant?
Answer: Saif al-Islam
Question: Who is he related to?
Answer: Moammar Gadhafi
Question: Who issued the arrest?
Answer: International Criminal Court
Question: What is the NTC?
Answer: National Transitional Counci
Question: Who spoke on behalf of them?
Answer: Luis Moreno-Ocampo
Question: Where does the NTC want the trial to be?
Answer: where he is captured
Question: Will he have rights if he goes to trial?
Answer: yes
Question: Who was the prosecutor speaking to?
Answer: CNN
Question: Where?
Answer: The Hague
text_B: Does he want him to go free?</pre>

**Question**

<pre>Answer the current question in text_B using the passage and earlier dialogue in text_A.</pre>

**Criteria (choice)**

<pre>1. no
2. yes</pre>

**Gold target (weight)**

<pre>1. no — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · coqa:train:67116e6f78b09f5e3ea67bc12d616d0f3e0679e1e86250d332e2c4b478157484:4:coqa-yesno-6b0f120687

**State**

<pre>text_A: CHAPTER XXI. BLACK STAR AND NIGHT&#32;

The time had come for Venters and Bess to leave their retreat. They were at great pains to choose the few things they would be able to carry with them on the journey out of Utah.&#32;

&quot;Bern, whatever kind of a pack&#x27;s this, anyhow?&quot; questioned Bess, rising from her work with reddened face.&#32;

Venters, absorbed in his own task, did not look up at all, and in reply said he had brought so much from Cottonwoods that he did not recollect the half of it.&#32;

&quot;A woman packed this!&quot; Bess exclaimed.&#32;

He scarcely caught her meaning, but the peculiar tone of her voice caused him instantly to rise, and he saw Bess on her knees before an open pack which he recognized as the one given him by Jane.&#32;

&quot;By George!&quot; he ejaculated, guiltily, and then at sight of Bess&#x27;s face he laughed outright.&#32;

&quot;A woman packed this,&quot; she repeated, fixing woeful, tragic eyes on him.&#32;

&quot;Well, is that a crime?&#x27;&#32;

&quot;There--there is a woman, after all!&quot;&#32;

&quot;Now Bess--&quot;&#32;

&quot;You&#x27;ve lied to me!&quot;&#32;

Then and there Venters found it imperative to postpone work for the present. All her life Bess had been isolated, but she had inherited certain elements of the eternal feminine.&#32;

&quot;But there was a woman and you did lie to me,&quot; she kept repeating, after he had explained.&#32;

&quot;What of that? Bess, I&#x27;ll get angry at you in a moment. Remember you&#x27;ve been pent up all your life. I venture to say that if you&#x27;d been out in the world you d have had a dozen sweethearts and have told many a lie before this.&quot;&#32;
Earlier dialogue:
Question: Where were Venters and Bess going to leave?
Answer: their retreat
Question: Where was it?
Answer: Utah.
Question: Could they take everything with them?
Answer: No
Question: Did Venters pack light?
Answer: Yes
text_B: Did a man pack the bag?</pre>

**Question**

<pre>Answer the current question in text_B using the passage and earlier dialogue in text_A.</pre>

**Criteria (choice)**

<pre>1. no
2. yes</pre>

**Gold target (weight)**

<pre>1. no — 1</pre>

</details>
