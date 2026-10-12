# esci

Classification · `tasksource/esci`

<details><summary>Example 1</summary>

published Jev · train · esci-c75e7e5ea0:train:15

**State**

<pre>text_A: tamaris shoes women
text_B: Tamaris Women&#x27;s Freya 24620 Slipper, Taupe Combo, 41 Medium EU (10 US)
Tamaris
Taupe Combo
Removable leather sock
TOUCH-IT padded insole</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. Complement
2. Exact
3. Irrelevant
4. Substitute</pre>

**Gold target (weight)**

<pre>2. Exact — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · esci-c75e7e5ea0:train:15:choice-paired-text-format

**State**

<pre>Passage A:
tamaris shoes women

Passage B:
Tamaris Women&#x27;s Freya 24620 Slipper, Taupe Combo, 41 Medium EU (10 US)
Tamaris
Taupe Combo
Removable leather sock
TOUCH-IT padded insole</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. Complement
2. Exact
3. Irrelevant
4. Substitute</pre>

**Gold target (weight)**

<pre>2. Exact — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · esci-c75e7e5ea0:train:1486

**State**

<pre>First text:
4xlt mens shirts big and tall

Second text:
Hanes Men&#x27;s Essentials Short Sleeve T-shirt Value Pack (4-pack),black,4X-Large
Hanes
Black
Experience real comfort with the Hanes ComfortSoft Tee.
Your Best Value in Comfort
100% Cotton (Ash 99% Cotton, 1% Polyester; Light Steel 90% Cotton, 10% Polyester; Charcoal Heather 50% Cotton, 50% Polyester)
High-quality comfort at an affordable price
Full cut provides roomier fit
Double stitched sleeves &amp; bottom hem for durability</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. Complement
2. Exact
3. Irrelevant
4. Substitute</pre>

**Gold target (weight)**

<pre>2. Exact — 1</pre>

</details>
