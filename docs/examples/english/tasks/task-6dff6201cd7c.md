# infotabs

Classification · `table-benchmark/infotabs`

<details><summary>Example 1</summary>

published Jev · train · infotabs:train:4419:infotabs-ca1437a260

**State**

<pre>text_A: Denver, Colorado
{&#x27;title&#x27;: [&#x27;Denver, Colorado&#x27;], &#x27;State capital and consolidated city-county&#x27;: [&#x27;154.97 sq mi (401.36 km 2 )&#x27;], &#x27;Land&#x27;: [&#x27;153.33 sq mi (397.13 km 2 )&#x27;], &#x27;Water&#x27;: [&#x27;1.63 sq mi (4.23 km 2 ) 1.1%&#x27;], &#x27;Metro&#x27;: [&#x27;8,414.4 sq mi (21,793 km 2 )&#x27;], &#x27;Elevation&#x27;: [&#x27;5,130-5,690 ft (1,564-1,734 m)&#x27;]}
text_B: Denver is a city located over a mile above sea level in Colorado.</pre>

**Question**

<pre>Does table text_A entail statement text_B, contradict it, or leave it undetermined?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · infotabs:train:5664:infotabs-ca1437a260

**State**

<pre>text_A: Charles
{&#x27;title&#x27;: [&#x27;Charles&#x27;], &#x27;Born&#x27;: [&#x27;(1948-11-14) 14 November 1948 (age 70) , Buckingham Palace, London, England&#x27;], &#x27;Spouse&#x27;: [&#x27;Lady Diana Spencer , ( m. 1981;  div. 1996)&#x27;, &#x27;Camilla Parker Bowles ,  ( m. 2005)&#x27;], &#x27;Issue&#x27;: [&#x27;Prince William, Duke of Cambridge&#x27;, &#x27;Prince Harry, Duke of Sussex&#x27;]}
text_B: Charles is an English prince who was married to the late Princess Diana.</pre>

**Question**

<pre>Does table text_A entail statement text_B, contradict it, or leave it undetermined?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · infotabs:train:897:infotabs-ca1437a260

**State**

<pre>text_A: Joseph Stalin
{&#x27;title&#x27;: [&#x27;Joseph Stalin&#x27;], &#x27;Nickname(s)&#x27;: [&#x27;Koba&#x27;], &#x27;Service / branch&#x27;: [&#x27;Soviet Armed Forces&#x27;], &#x27;Years of service&#x27;: [&#x27;1943 - 1953&#x27;], &#x27;Rank&#x27;: [&#x27;Marshal of the Soviet Union (1943 - 1945)&#x27;], &#x27;Battles/wars&#x27;: [&#x27;World War II&#x27;, &#x27;Russian Civil War&#x27;, &#x27;Polish-Soviet War&#x27;, &#x27;Winter War&#x27;], &#x27;Central institution membership&#x27;: [&#x27;1917–53: Full member, 6th–19th Presidium&#x27;, &#x27;1922–43: 11th–19th Secretariat&#x27;, &#x27;1920–52: 9th–18th Orgburo&#x27;, &#x27;1912–53: Full member&#x27;, &#x27; 5th–19th Central Committee&#x27;], &#x27;Other offices held&#x27;: [&#x27;1941–47: Minister of Defense&#x27;, &#x27; 1941–45: Chairman, State Defense Committee&#x27;, &quot;  1920–22: Chairman, Workers&#x27; and Peasants&#x27; Inspectorate of the Russian SFSR &quot;, &quot; 1917–22: People&#x27;s Commissar, Nationalities of the Russian SFSR&quot;, &#x27;Leader of the Soviet Union&#x27;]}
text_B: Joseph Stalin wound up staying in the army into the cold war.</pre>

**Question**

<pre>Does table text_A entail statement text_B, contradict it, or leave it undetermined?</pre>

**Criteria (choice)**

<pre>1. entailment
2. contradiction
3. neutral</pre>

**Gold target (weight)**

<pre>2. contradiction — 1</pre>

</details>
