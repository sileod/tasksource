# clcd-english

Classification · `tasksource/clcd-english`

<details><summary>Example 1</summary>

published Jev · train · clcd-english-ee91b0e324:train:106

**State**

<pre>text_A: Leslie is as tall as Raul , Raul is as tall as Jon , Jon is as tall as Gerald , Gerald is as tall as Tom , Tom is as tall as Jimmy , Jimmy is as tall as Kenneth , Kenneth is as tall as Seth , Seth is as tall as Lester , Lester is as tall as Wallace , Wallace is as tall as Rodney , Rodney is taller than Eric
text_B: Eric is taller than Lester</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. contradiction
2. not_contradiction</pre>

**Gold target (weight)**

<pre>1. contradiction — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · clcd-english-ee91b0e324:train:106:choice-paired-text-format

**State**

<pre>First text:
Leslie is as tall as Raul , Raul is as tall as Jon , Jon is as tall as Gerald , Gerald is as tall as Tom , Tom is as tall as Jimmy , Jimmy is as tall as Kenneth , Kenneth is as tall as Seth , Seth is as tall as Lester , Lester is as tall as Wallace , Wallace is as tall as Rodney , Rodney is taller than Eric

Second text:
Eric is taller than Lester</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. contradiction
2. not_contradiction</pre>

**Gold target (weight)**

<pre>1. contradiction — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · clcd-english-ee91b0e324:train:4495

**State**

<pre>First text:
George has visited Nepal, Benin, Republic of the Congo, Belize, Uganda, Kuwait, Sweden, Mexico, Jordan, Venezuela, Gabon, Burundi, Tunisia, Malaysia, Paraguay, Bolivia, Chad, Dominica and Grenada

Second text:
George didn&#x27;t visit Syria</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. contradiction
2. not_contradiction</pre>

**Gold target (weight)**

<pre>2. not_contradiction — 1</pre>

</details>
