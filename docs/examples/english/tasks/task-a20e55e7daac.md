# dataset_train_nli

Classification · `MoritzLaurer/dataset_train_nli`

<details><summary>Example 1</summary>

published Jev · train · dataset-train-nli-91e59a492d:train:pack-188107ea57f9:label-B

**State**

<pre>Item A:
text_A: bodoi.info - information about domain
- Server Status: available
- Server IP: 213.186.33.17
- Server Response Time: 96ms
Our site strives to provide helpful information to our readers. By offering not only fact-based whois information, but also an informative overview of each website, we seek to give you a broad view of how each site operates. Every day our writers and researchers work together to find all of the most up-to-date information about each website we feature. Our goal is to create a space where you will gain the most information for your time, because in our view, our time is spent saving others’ time. Rather than spending hours scouring the web for the information you seek, you will be able to find it all in one well-organized space, allowing you more time to enjoy browsing the web at your leisure.
This site has been a great resource for me. The ease of use, wide variety of information, as well as the great layout are a refreshing change to the standard, bulky sites I have used in the past. Here I am able to find everything I need in one site, without spending hours sifting through little bits of information across the Internet. This has led to improving my own productivity, because I spend far less time searching all over the place for the content I need. I definitely see a strong future for this site and will continue to use it for my needs.
eRic000 - eRic Design
text_B: This text is about: gaming

Item B:
text_A: Puente Hills Nissan has a great selection of new 2013 Nissan cars, vans, SUVs and trucks in the area of Los Angeles County areas including: West Covina, La Puente, Hacienda Heights, Whittier, Rowland Heights, Diamond Bar, El Monte, Walnut, La Habra, Pomona, Baldwin Park, Chino Hills, Yorba Linda, Rosemead, Downey, Monterey Park and surrounding Southern California communities. We also have a nice selection of new 2012 Nissan models and the new 2014 Nissan GT-R in stock now. See hundreds of new Nissan cars, SUVs and trucks for sale including the: Nissan Altima, Versa, Sentra, Maxima, Juke, LEAF, Frontier, Pathfinder, Armada, 370Z, Cube, Murano, Murano CrossCabriolet, Quest, Rogue, Titan, Xterra, and Nissan NV - Los Angeles Area.
text_B: This text is about: car parts</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>2. not_entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · dataset-train-nli-91e59a492d:train:pack-188107ea57f9:same-A-B

**State**

<pre>Item A:
text_A: bodoi.info - information about domain
- Server Status: available
- Server IP: 213.186.33.17
- Server Response Time: 96ms
Our site strives to provide helpful information to our readers. By offering not only fact-based whois information, but also an informative overview of each website, we seek to give you a broad view of how each site operates. Every day our writers and researchers work together to find all of the most up-to-date information about each website we feature. Our goal is to create a space where you will gain the most information for your time, because in our view, our time is spent saving others’ time. Rather than spending hours scouring the web for the information you seek, you will be able to find it all in one well-organized space, allowing you more time to enjoy browsing the web at your leisure.
This site has been a great resource for me. The ease of use, wide variety of information, as well as the great layout are a refreshing change to the standard, bulky sites I have used in the past. Here I am able to find everything I need in one site, without spending hours sifting through little bits of information across the Internet. This has led to improving my own productivity, because I spend far less time searching all over the place for the content I need. I definitely see a strong future for this site and will continue to use it for my needs.
eRic000 - eRic Design
text_B: This text is about: gaming

Item B:
text_A: Puente Hills Nissan has a great selection of new 2013 Nissan cars, vans, SUVs and trucks in the area of Los Angeles County areas including: West Covina, La Puente, Hacienda Heights, Whittier, Rowland Heights, Diamond Bar, El Monte, Walnut, La Habra, Pomona, Baldwin Park, Chino Hills, Yorba Linda, Rosemead, Downey, Monterey Park and surrounding Southern California communities. We also have a nice selection of new 2012 Nissan models and the new 2014 Nissan GT-R in stock now. See hundreds of new Nissan cars, SUVs and trucks for sale including the: Nissan Altima, Versa, Sentra, Maxima, Juke, LEAF, Frontier, Pathfinder, Armada, 370Z, Cube, Murano, Murano CrossCabriolet, Quest, Rogue, Titan, Xterra, and Nissan NV - Los Angeles Area.
text_B: This text is about: car parts</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Do Item A and Item B have the same label? Possible labels: &quot;entailment&quot;, &quot;not_entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · dataset-train-nli-91e59a492d:train:pack-188107ea57f9:exists-1

**State**

<pre>Item A:
text_A: bodoi.info - information about domain
- Server Status: available
- Server IP: 213.186.33.17
- Server Response Time: 96ms
Our site strives to provide helpful information to our readers. By offering not only fact-based whois information, but also an informative overview of each website, we seek to give you a broad view of how each site operates. Every day our writers and researchers work together to find all of the most up-to-date information about each website we feature. Our goal is to create a space where you will gain the most information for your time, because in our view, our time is spent saving others’ time. Rather than spending hours scouring the web for the information you seek, you will be able to find it all in one well-organized space, allowing you more time to enjoy browsing the web at your leisure.
This site has been a great resource for me. The ease of use, wide variety of information, as well as the great layout are a refreshing change to the standard, bulky sites I have used in the past. Here I am able to find everything I need in one site, without spending hours sifting through little bits of information across the Internet. This has led to improving my own productivity, because I spend far less time searching all over the place for the content I need. I definitely see a strong future for this site and will continue to use it for my needs.
eRic000 - eRic Design
text_B: This text is about: gaming

Item B:
text_A: Puente Hills Nissan has a great selection of new 2013 Nissan cars, vans, SUVs and trucks in the area of Los Angeles County areas including: West Covina, La Puente, Hacienda Heights, Whittier, Rowland Heights, Diamond Bar, El Monte, Walnut, La Habra, Pomona, Baldwin Park, Chino Hills, Yorba Linda, Rosemead, Downey, Monterey Park and surrounding Southern California communities. We also have a nice selection of new 2012 Nissan models and the new 2014 Nissan GT-R in stock now. See hundreds of new Nissan cars, SUVs and trucks for sale including the: Nissan Altima, Versa, Sentra, Maxima, Juke, LEAF, Frontier, Pathfinder, Armada, 370Z, Cube, Murano, Murano CrossCabriolet, Quest, Rogue, Titan, Xterra, and Nissan NV - Los Angeles Area.
text_B: This text is about: car parts</pre>

**Question**

<pre>Each item answers: &quot;Does text_A entail text_B?&quot;
Does at least one item have the label &quot;not_entailment&quot;? Possible labels: &quot;entailment&quot;, &quot;not_entailment&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
