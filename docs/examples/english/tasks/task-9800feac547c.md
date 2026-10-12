# amazon_polarity/amazon_polarity

Classification · `fancyzhx/amazon_polarity`

<details><summary>Example 1</summary>

published Jev · train · amazon-polarity-amazon-polarity-e566d970d0:train:pack-c26f1218e48d:label-A

**State**

<pre>Item A:
Gore-hounds take note, &quot;The Gore Gore Girls&quot; is perhaps the most gruesome film made by Herschell Gordon Lewis. Of course, the viewer gets to witness some rather trashy strip sequences that contain more scares than the movie itself! Everything else is pretty standard: gross effects, bad acting (though there are a few surprises here, such as a guest appearance by &quot;The King Of the One-Liners&quot; Henny Youngman), and some absolutely awful directing! For the most part, I felt that this film was a meager step up from earlier Lewis films like &quot;The Gruesome Twosome&quot;, and the jokes actually seemed more intentional this time!The DVD is a well-done project, featuring a wonderfully candid HGL interview/commentary that tells you everything you wanted to know, or didn&#x27;t want to know, about &quot;The Gore Gore Girls&quot;. The other extras, especially the Goddess Of Gore segment, are also interesting for early-gore film fanatics.

Item B:
Tried to save some money by buying these online. Amazon sent me 2 sets of passenger side brake pads and none for the drivers side. Dont buy these pads.</pre>

**Question**

<pre>Each item answers: &quot;What sentiment does the text express?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. negative
2. positive</pre>

**Gold target (weight)**

<pre>1. negative — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · amazon-polarity-amazon-polarity-e566d970d0:train:pack-c26f1218e48d:in-A-1

**State**

<pre>Item A:
Gore-hounds take note, &quot;The Gore Gore Girls&quot; is perhaps the most gruesome film made by Herschell Gordon Lewis. Of course, the viewer gets to witness some rather trashy strip sequences that contain more scares than the movie itself! Everything else is pretty standard: gross effects, bad acting (though there are a few surprises here, such as a guest appearance by &quot;The King Of the One-Liners&quot; Henny Youngman), and some absolutely awful directing! For the most part, I felt that this film was a meager step up from earlier Lewis films like &quot;The Gruesome Twosome&quot;, and the jokes actually seemed more intentional this time!The DVD is a well-done project, featuring a wonderfully candid HGL interview/commentary that tells you everything you wanted to know, or didn&#x27;t want to know, about &quot;The Gore Gore Girls&quot;. The other extras, especially the Goddess Of Gore segment, are also interesting for early-gore film fanatics.

Item B:
Tried to save some money by buying these online. Amazon sent me 2 sets of passenger side brake pads and none for the drivers side. Dont buy these pads.</pre>

**Question**

<pre>Each item answers: &quot;What sentiment does the text express?&quot;
Is the label of Item A &quot;positive&quot;? Possible labels: &quot;negative&quot;, &quot;positive&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · amazon-polarity-amazon-polarity-e566d970d0:train:pack-c26f1218e48d:all-same

**State**

<pre>Item A:
Gore-hounds take note, &quot;The Gore Gore Girls&quot; is perhaps the most gruesome film made by Herschell Gordon Lewis. Of course, the viewer gets to witness some rather trashy strip sequences that contain more scares than the movie itself! Everything else is pretty standard: gross effects, bad acting (though there are a few surprises here, such as a guest appearance by &quot;The King Of the One-Liners&quot; Henny Youngman), and some absolutely awful directing! For the most part, I felt that this film was a meager step up from earlier Lewis films like &quot;The Gruesome Twosome&quot;, and the jokes actually seemed more intentional this time!The DVD is a well-done project, featuring a wonderfully candid HGL interview/commentary that tells you everything you wanted to know, or didn&#x27;t want to know, about &quot;The Gore Gore Girls&quot;. The other extras, especially the Goddess Of Gore segment, are also interesting for early-gore film fanatics.

Item B:
Tried to save some money by buying these online. Amazon sent me 2 sets of passenger side brake pads and none for the drivers side. Dont buy these pads.</pre>

**Question**

<pre>Each item answers: &quot;What sentiment does the text express?&quot;
Do all items have the same label? Possible labels: &quot;negative&quot;, &quot;positive&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
