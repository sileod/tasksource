# multidogo/software/intent

Classification · `jpcorb20/multidogo`

<details><summary>Example 1</summary>

current local Jev recast (not yet published) · train · multidogo-software-intent-7b749275b2:train:0

**State**

<pre>text_A: Turn 1: hello good morning
Turn 2: my skpye not working
Turn 4: prakash
Turn 5: 2583
Turn 7: prakash info
Turn 9: okay
Turn 11: yes
text_B: no thank you</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. confirmation
2. contentonly
3. expensereport
4. rejection
5. softwareupdate
6. thankyou</pre>

**Gold target (weight)**

<pre>4. rejection — 1</pre>

</details>

<details><summary>Example 2</summary>

current local Jev recast (not yet published) · train · multidogo-software-intent-7b749275b2:train:1

**State**

<pre>text_A: Turn 0: hai
Turn 2: i want new software update
text_B: camile 8787</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. confirmation
2. contentonly
3. expensereport
4. rejection
5. softwareupdate
6. thankyou</pre>

**Gold target (weight)**

<pre>2. contentonly — 1</pre>

</details>

<details><summary>Example 3</summary>

current local Jev recast (not yet published) · train · multidogo-software-intent-7b749275b2:train:2

**State**

<pre>text_A: Turn 0: hi
text_B: i want to update my skype</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. confirmation
2. contentonly
3. expensereport
4. rejection
5. softwareupdate
6. thankyou</pre>

**Gold target (weight)**

<pre>5. softwareupdate — 1</pre>

</details>
