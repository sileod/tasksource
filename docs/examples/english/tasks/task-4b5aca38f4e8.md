# cosmos_qa

MultipleChoice · `Samsoup/cosmos_qa`

<details><summary>Example 1</summary>

published Jev · train · cosmos-qa-81b1d873b1:train:67

**State**

<pre>We stopped by Felix &#x27;s grave for a few minutes on the way there . I have only been there once before ( a few years ago ) , so it made me happy that I got to go there again . The parade was just okay . It was mostly a ton of people walking ( little league , boy scouts , girl scouts , etc ) , so there was n&#x27;t much to see . What may have caused you to stop on your way to the parade ?</pre>

**Question**

<pre>Which supplied option best answers the question?</pre>

**Criteria (choice)**

<pre>1. I wanted to pay my respects to someone who &#x27;s passed one .
2. Someone asked if they could pay their respects to me .
3. I wanted to hang out with the boy scouts .
4. I wanted to pay Felix a visit and see if he would go to the parade with me .</pre>

**Gold target (weight)**

<pre>1. I wanted to pay my respects to someone who &#x27;s passed one . — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · cosmos-qa-81b1d873b1:train:67:choice-criteria-permutation

**State**

<pre>We stopped by Felix &#x27;s grave for a few minutes on the way there . I have only been there once before ( a few years ago ) , so it made me happy that I got to go there again . The parade was just okay . It was mostly a ton of people walking ( little league , boy scouts , girl scouts , etc ) , so there was n&#x27;t much to see . What may have caused you to stop on your way to the parade ?</pre>

**Question**

<pre>Which supplied option best answers the question?</pre>

**Criteria (choice)**

<pre>1. I wanted to hang out with the boy scouts .
2. I wanted to pay Felix a visit and see if he would go to the parade with me .
3. I wanted to pay my respects to someone who &#x27;s passed one .
4. Someone asked if they could pay their respects to me .</pre>

**Gold target (weight)**

<pre>3. I wanted to pay my respects to someone who &#x27;s passed one . — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · cosmos-qa-81b1d873b1:train:5124

**State**

<pre>They were nothing but helpful and polite , such a nice surprise . I &#x27; m going back today to place product and do finishing touches . I &#x27;ll be back next week - hopefully with stories ! Have a great weekend ! What is the reason for placing product ?</pre>

**Question**

<pre>Choose the most appropriate answer from the supplied options.</pre>

**Criteria (choice)**

<pre>1. None of the other options .
2. The product is invisible ink .
3. The product is saleable items for display and sale .
4. The product are christmas trees .</pre>

**Gold target (weight)**

<pre>3. The product is saleable items for display and sale . — 1</pre>

</details>
