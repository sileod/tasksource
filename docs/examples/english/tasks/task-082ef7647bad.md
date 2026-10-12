# conceptrules_v2

Classification · `tasksource/conceptrules_v2`

<details><summary>Example 1</summary>

published Jev · train · conceptrules-v2-83f331d8b8:train:76

**State**

<pre>text_A: Pizza is in the vicinity of plate. Plate is in the vicinity of dishwasher. Weasel is not in the vicinity of your eye. Plate is used for put food on. Weasel is in the vicinity of cheese. Cheese is in the vicinity of pizza. Weasel is not in the vicinity of book. Lettuce is not in the vicinity of populous area. Cheese is not in the vicinity of your eye. Dentist chair is in the vicinity of populous area. Lettuce is in the vicinity of dishwasher.
text_B: Plate is not in the vicinity of dishwasher.</pre>

**Question**

<pre>Is the statement true given the context?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · conceptrules-v2-83f331d8b8:train:76:choice-paired-text-format

**State**

<pre>Passage A:
Pizza is in the vicinity of plate. Plate is in the vicinity of dishwasher. Weasel is not in the vicinity of your eye. Plate is used for put food on. Weasel is in the vicinity of cheese. Cheese is in the vicinity of pizza. Weasel is not in the vicinity of book. Lettuce is not in the vicinity of populous area. Cheese is not in the vicinity of your eye. Dentist chair is in the vicinity of populous area. Lettuce is in the vicinity of dishwasher.

Passage B:
Plate is not in the vicinity of dishwasher.</pre>

**Question**

<pre>Is the statement true given the context?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · conceptrules-v2-83f331d8b8:train:6278

**State**

<pre>Passage A:
Bedroom is not in the vicinity of sweetshop. House is in the vicinity of subdivision. News is not in the vicinity of this tropical island. News is in the vicinity of tv. News is not in the vicinity of sweetshop. Jeep is not in the vicinity of pencil box. Jeep is not in the vicinity of subdivision. Satelite photo is in the vicinity of pencil box. Bedroom is in the vicinity of house. Tv is in the vicinity of bedroom.

Passage B:
Tv is not in the vicinity of bedroom.</pre>

**Question**

<pre>Is the statement true given the context?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>
