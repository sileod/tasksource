# multidogo/airline/intent

Classification · `jpcorb20/multidogo`

<details><summary>Example 1</summary>

current local Jev recast (not yet published) · train · multidogo-airline-intent-b02020c0b4:train:0

**State**

<pre>text_A: Turn 0: hai
Turn 2: check my seat assignment
text_B: df 3 s 22</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. bookflight
2. confirmation
3. contentonly
4. openinggreeting
5. thankyou</pre>

**Gold target (weight)**

<pre>3. contentonly — 1</pre>

</details>

<details><summary>Example 2</summary>

current local Jev recast (not yet published) · train · multidogo-airline-intent-b02020c0b4:train:1

**State**

<pre>hi</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. bookflight
2. confirmation
3. contentonly
4. openinggreeting
5. thankyou</pre>

**Gold target (weight)**

<pre>4. openinggreeting — 1</pre>

</details>

<details><summary>Example 3</summary>

current local Jev recast (not yet published) · train · multidogo-airline-intent-b02020c0b4:train:2

**State**

<pre>text_A: Turn 1: hii
Turn 2: i need to change my seat assignment
Turn 4: subramani
Turn 6: rvt 458
Turn 8: window seat
Turn 10: ok thanks
text_B: ok thanks</pre>

**Question**

<pre>What intent does the current (last) utterance express? Use earlier dialogue when available.</pre>

**Criteria (choice)**

<pre>1. bookflight
2. confirmation
3. contentonly
4. openinggreeting
5. thankyou</pre>

**Gold target (weight)**

<pre>5. thankyou — 1</pre>

</details>
