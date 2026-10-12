# counterfactually-augmented-imdb

Classification · `tasksource/counterfactually-augmented-imdb`

<details><summary>Example 1</summary>

published Jev · train · counterfactually-augmented-imdb-72396a9ef6:train:0

**State**

<pre>I was never quite sure where this thing was going. These people seem interested in what is going on on some mountain. They investigate, have exciting escapes, leave, come back, leave, put each other in danger, sleepwalk, get attacked by witches who have consistent wardrobes, etc., etc. The guy seems to like the girl, but leaves her unprotected numerous times. She gets taken off, he gets her back, leaves her again. You get immersed in the film. The whole thing seems to get around to some sort of sacrifice, I think, but I&#x27;m not sure, or turning people into witches, but I&#x27;m not sure. It&#x27;s just enthralling  and definitely worth the time. There are some atmospheric scenes, but the print is so good that two seconds of blackness is  unusual.</pre>

**Question**

<pre>What sentiment does the text express?</pre>

**Criteria (choice)**

<pre>1. Negative
2. Positive</pre>

**Gold target (weight)**

<pre>2. Positive — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · counterfactually-augmented-imdb-72396a9ef6:train:0:noul-label-verification

**State**

<pre>I was never quite sure where this thing was going. These people seem interested in what is going on on some mountain. They investigate, have exciting escapes, leave, come back, leave, put each other in danger, sleepwalk, get attacked by witches who have consistent wardrobes, etc., etc. The guy seems to like the girl, but leaves her unprotected numerous times. She gets taken off, he gets her back, leaves her again. You get immersed in the film. The whole thing seems to get around to some sort of sacrifice, I think, but I&#x27;m not sure, or turning people into witches, but I&#x27;m not sure. It&#x27;s just enthralling  and definitely worth the time. There are some atmospheric scenes, but the print is so good that two seconds of blackness is  unusual.</pre>

**Question**

<pre>What sentiment does the text express? Is &quot;Positive&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · counterfactually-augmented-imdb-72396a9ef6:train:285

**State**

<pre>It has been 16 years since it&#x27;s original run, I would have hoped by now some &quot;marketing wizard&quot; would have promoted a live actor version of this classic by now, or at least sought to re-release the original 65 episodes. I can&#x27;t fathom why the sci-fi or cartoon network haven&#x27;t snapped this up. Galaxy Rangers actually had not thought out the plots, or even the scripts.The animation was below average quality for it&#x27;s time, and bad when compared to the talking slide show Japanese animation of today. It predated the heavy toon-toy tie in market, this may have sealed it&#x27;s doom too. I would not willingly spend cash on a DVD of GR if available.</pre>

**Question**

<pre>What sentiment does the text express?</pre>

**Criteria (choice)**

<pre>1. Negative
2. Positive</pre>

**Gold target (weight)**

<pre>1. Negative — 1</pre>

</details>
