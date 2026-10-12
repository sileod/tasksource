# HelpSteer/correctness

Classification · `nvidia/HelpSteer`

<details><summary>Example 1</summary>

published Jev · train · HelpSteer-correctness-b6bde711ef:train:2093

**State**

<pre>Passage A:
Refer to the information below to help with the following  delimited in ###:
###
On May 12, 2001, Shrek improbably premiered at the Cannes Film Festival, becoming the first animated movie to compete for the Palme d’Or since Disney’s Peter Pan in 1953. It was, on the surface, an extreme mismatch between movie and moviegoers. Imagine the moment: Throngs of big-screen stars, top-flight filmmakers, and cinema cognoscenti are assembled in black tie to watch the world’s finest films. Then an animated ogre wipes his ass, emerges from an outhouse, and adjusts a wedgie to the strains of Smash Mouth’s “All Star.”

“During that whole opening sequence, as Smash Mouth is playing and Shrek’s farting in the mud pool, I was just sweating,” says Andrew Adamson, who codirected Shrek with Vicky Jenson and went on to codirect Shrek 2. “I said, ‘This is just going to be a disaster. This is not the right audience for this film. … The song, the animation, everything about this was a poor choice to be bringing here.’”

Shrek lost out to Italian film The Son’s Room in the fight for the Palme d’Or. But six days after its Cannes premiere, thousands of theaters got the show on and got paid. Shrek grossed almost $500 million worldwide—more than 41 times as much as The Son’s Room—and spawned three sequels and a musical, plus a Puss in Boots spinoff film. Beloved by both critics and the public, the original Shrek won the Academy Award for Best Animated Feature, and Shrek 2 opened Cannes in 2004.

The Shrek franchise has been dormant for a decade, but 20 years after Shrek’s theatrical release, producer DreamWorks is stirring the IP pot: Shrek is about to be rebooted, and Puss in Boots is slated to receive a sequel of its own. Even aside from those scheduled new installments, though, Shrek casts an ogre-sized shadow over 21st-century animated movies. The success of the original and its even more lucrative sequels started and sustained a trend toward parodic, crude, and pop-culture-packed takes on fairy tales. As much as Shrek thrived on outhouse humor, postmodern deconstructions of Disney-style fables, and the star power of a cast that included Mike Myers, Eddie Murphy, Cameron Diaz, and John Lithgow, its signature contribution to the culture—and the main fuel for nostalgia among millennials—may be its boundary-smashing use of licensed popular songs, especially “All Star,” “I’m a Believer,” and “Hallelujah.” Shrek’s Smash Mouth–suffused soundtrack climbed to the top of the Billboard soundtrack chart and went multiplatinum, and the soundtrack to Shrek 2 hit no. 1 and went platinum too.

“Shrek was the first animated picture to have, in its dialogue and in its music, pop references,” says Marylata Elton, Shrek’s music supervisor. “What ended up [happening] is that Shrek became pop culture itself.”
###

Imagine you&#x27;re trying to promote the movie Shrek. Using the article above, create a script for a radio ad. Your goal is to convince people to come and watch Shrek. 
As part of the script, include a 4-line stanza rewrite of the popular smash mouth song with the lyrics being from the POV of Shrek.

Passage B:
Welcome to another exciting film premier of Shrek!
The most loved ogre of all time!
This smash mouth hit song is from the
POV of our favorite green guy!

Shrek had no choice but to say no
as the parade went by
he just wanted some privacy
in his swamp

But the band was too loud
and it was hard to think
he couldn&#x27;t take it anymore
and started to sing

&quot;Hello, world! I&#x27;m a Believer!&quot;
that&#x27;s what he screamed
in a very loud voice
that echoed and roared

It&#x27;s not like he didn&#x27;t want to be friendly
he just didn&#x27;t want to be known
as the ogre who
kept singing this song</pre>

**Question**

<pre>How would you rate the correctness of the response?</pre>

**Criteria (choice)**

<pre>1. 0: mostly incorrect
2. 1
3. 2
4. 3
5. 4: fully correct and complete</pre>

**Gold target (weight)**

<pre>3. 2 — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · HelpSteer-correctness-b6bde711ef:train:2093:choice-paired-text-format

**State**

<pre>A: Refer to the information below to help with the following  delimited in ###:
###
On May 12, 2001, Shrek improbably premiered at the Cannes Film Festival, becoming the first animated movie to compete for the Palme d’Or since Disney’s Peter Pan in 1953. It was, on the surface, an extreme mismatch between movie and moviegoers. Imagine the moment: Throngs of big-screen stars, top-flight filmmakers, and cinema cognoscenti are assembled in black tie to watch the world’s finest films. Then an animated ogre wipes his ass, emerges from an outhouse, and adjusts a wedgie to the strains of Smash Mouth’s “All Star.”

“During that whole opening sequence, as Smash Mouth is playing and Shrek’s farting in the mud pool, I was just sweating,” says Andrew Adamson, who codirected Shrek with Vicky Jenson and went on to codirect Shrek 2. “I said, ‘This is just going to be a disaster. This is not the right audience for this film. … The song, the animation, everything about this was a poor choice to be bringing here.’”

Shrek lost out to Italian film The Son’s Room in the fight for the Palme d’Or. But six days after its Cannes premiere, thousands of theaters got the show on and got paid. Shrek grossed almost $500 million worldwide—more than 41 times as much as The Son’s Room—and spawned three sequels and a musical, plus a Puss in Boots spinoff film. Beloved by both critics and the public, the original Shrek won the Academy Award for Best Animated Feature, and Shrek 2 opened Cannes in 2004.

The Shrek franchise has been dormant for a decade, but 20 years after Shrek’s theatrical release, producer DreamWorks is stirring the IP pot: Shrek is about to be rebooted, and Puss in Boots is slated to receive a sequel of its own. Even aside from those scheduled new installments, though, Shrek casts an ogre-sized shadow over 21st-century animated movies. The success of the original and its even more lucrative sequels started and sustained a trend toward parodic, crude, and pop-culture-packed takes on fairy tales. As much as Shrek thrived on outhouse humor, postmodern deconstructions of Disney-style fables, and the star power of a cast that included Mike Myers, Eddie Murphy, Cameron Diaz, and John Lithgow, its signature contribution to the culture—and the main fuel for nostalgia among millennials—may be its boundary-smashing use of licensed popular songs, especially “All Star,” “I’m a Believer,” and “Hallelujah.” Shrek’s Smash Mouth–suffused soundtrack climbed to the top of the Billboard soundtrack chart and went multiplatinum, and the soundtrack to Shrek 2 hit no. 1 and went platinum too.

“Shrek was the first animated picture to have, in its dialogue and in its music, pop references,” says Marylata Elton, Shrek’s music supervisor. “What ended up [happening] is that Shrek became pop culture itself.”
###

Imagine you&#x27;re trying to promote the movie Shrek. Using the article above, create a script for a radio ad. Your goal is to convince people to come and watch Shrek. 
As part of the script, include a 4-line stanza rewrite of the popular smash mouth song with the lyrics being from the POV of Shrek.
B: Welcome to another exciting film premier of Shrek!
The most loved ogre of all time!
This smash mouth hit song is from the
POV of our favorite green guy!

Shrek had no choice but to say no
as the parade went by
he just wanted some privacy
in his swamp

But the band was too loud
and it was hard to think
he couldn&#x27;t take it anymore
and started to sing

&quot;Hello, world! I&#x27;m a Believer!&quot;
that&#x27;s what he screamed
in a very loud voice
that echoed and roared

It&#x27;s not like he didn&#x27;t want to be friendly
he just didn&#x27;t want to be known
as the ogre who
kept singing this song</pre>

**Question**

<pre>How would you rate the correctness of the response?</pre>

**Criteria (choice)**

<pre>1. 0: mostly incorrect
2. 1
3. 2
4. 3
5. 4: fully correct and complete</pre>

**Gold target (weight)**

<pre>3. 2 — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · HelpSteer-correctness-b6bde711ef:train:26759

**State**

<pre>Passage A:
Write a poem about a horse&#x27;s hoof from the horse&#x27;s perspective. Use the lines &quot;under my skin, in my bone&quot; in the poem.

Passage B:
Under my skin, in my bone,
A song that only I can hear,
From my head to my hooves,
I am a horse, running free.

With each gallop and each jump,
I feel the earth beneath my feet,
And in my heart, I know,
This is where I&#x27;m meant to be.

The wind blows through my mane,
And I am free, I am free,
To run and play and be,
The horse that I was meant to be.</pre>

**Question**

<pre>How would you rate the correctness of the response?</pre>

**Criteria (choice)**

<pre>1. 0: mostly incorrect
2. 1
3. 2
4. 3
5. 4: fully correct and complete</pre>

**Gold target (weight)**

<pre>5. 4: fully correct and complete — 1</pre>

</details>
