# FOL-nli

Classification · `unigram/FOL-nli`

<details><summary>Example 1</summary>

published Jev · train · FOL-nli-3deb6ff51f:train:6

**State**

<pre>text_A: Jeffrey, Jason, Earnest are the only persons in the room. Everyone in the room who is a Linux enthusiast enjoys coding in Python. Everyone in the room who enjoys coding in Python enjoys spelunking, watches fantasy movies or owns an Android phone. Jeffrey reads mystery novels. If someone enjoys spelunking then he/she enjoys coding in Python and vice versa. Everyone in the room enjoys origami only if they collects first edition books. Jeffrey creates augmented reality experiences for mobile applications. Earnest enjoys coding in Python. Someone in the room collects first edition books. Everyone in the room who is a Linux enthusiast practices urban gardening. Everyone in the room who practices urban gardening is a wine connoisseur and creates YouTube content. Only one person in the room enjoys coding in Python or enjoys origami or both. Everyone in the room who reads mystery novels neither creates YouTube content nor enjoys origami. Jason collects historical artifacts related to ancient civilizations. Everyone in the room who enjoys coding in Python is a wine connoisseur. If someone reads mystery novels then he/she collects historical artifacts related to ancient civilizations.
text_B: Everyone in the room is a Linux enthusiast.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · FOL-nli-3deb6ff51f:train:6:noul-label-verification

**State**

<pre>text_A: Jeffrey, Jason, Earnest are the only persons in the room. Everyone in the room who is a Linux enthusiast enjoys coding in Python. Everyone in the room who enjoys coding in Python enjoys spelunking, watches fantasy movies or owns an Android phone. Jeffrey reads mystery novels. If someone enjoys spelunking then he/she enjoys coding in Python and vice versa. Everyone in the room enjoys origami only if they collects first edition books. Jeffrey creates augmented reality experiences for mobile applications. Earnest enjoys coding in Python. Someone in the room collects first edition books. Everyone in the room who is a Linux enthusiast practices urban gardening. Everyone in the room who practices urban gardening is a wine connoisseur and creates YouTube content. Only one person in the room enjoys coding in Python or enjoys origami or both. Everyone in the room who reads mystery novels neither creates YouTube content nor enjoys origami. Jason collects historical artifacts related to ancient civilizations. Everyone in the room who enjoys coding in Python is a wine connoisseur. If someone reads mystery novels then he/she collects historical artifacts related to ancient civilizations.
text_B: Everyone in the room is a Linux enthusiast.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither? Is &quot;contradiction&quot; the correct answer?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · FOL-nli-3deb6ff51f:train:6:choice-paired-text-format

**State**

<pre>Passage A:
Jeffrey, Jason, Earnest are the only persons in the room. Everyone in the room who is a Linux enthusiast enjoys coding in Python. Everyone in the room who enjoys coding in Python enjoys spelunking, watches fantasy movies or owns an Android phone. Jeffrey reads mystery novels. If someone enjoys spelunking then he/she enjoys coding in Python and vice versa. Everyone in the room enjoys origami only if they collects first edition books. Jeffrey creates augmented reality experiences for mobile applications. Earnest enjoys coding in Python. Someone in the room collects first edition books. Everyone in the room who is a Linux enthusiast practices urban gardening. Everyone in the room who practices urban gardening is a wine connoisseur and creates YouTube content. Only one person in the room enjoys coding in Python or enjoys origami or both. Everyone in the room who reads mystery novels neither creates YouTube content nor enjoys origami. Jason collects historical artifacts related to ancient civilizations. Everyone in the room who enjoys coding in Python is a wine connoisseur. If someone reads mystery novels then he/she collects historical artifacts related to ancient civilizations.

Passage B:
Everyone in the room is a Linux enthusiast.</pre>

**Question**

<pre>Does text_A entail text_B, contradict it, or neither?</pre>

**Criteria (choice)**

<pre>1. entailment
2. neutral
3. contradiction</pre>

**Gold target (weight)**

<pre>3. contradiction — 1</pre>

</details>
