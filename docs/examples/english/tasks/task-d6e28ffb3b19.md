# cloth

MultipleChoice · `AndyChiang/cloth`

<details><summary>Example 1</summary>

published Jev · train · cloth-a8d3866ed4:train:4

**State**

<pre>He would [MASK] the basket on purpose  so I wouldn&#x27;t lose against him.</pre>

**Question**

<pre>Choose the most appropriate answer from the supplied options.</pre>

**Criteria (choice)**

<pre>1. miss
2. hit
3. catch
4. get</pre>

**Gold target (weight)**

<pre>1. miss — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · cloth-a8d3866ed4:train:4:noul-label-verification

**State**

<pre>He would [MASK] the basket on purpose  so I wouldn&#x27;t lose against him.</pre>

**Question**

<pre>Is &quot;miss&quot; the correct answer to the question?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · cloth-a8d3866ed4:train:14220

**State**

<pre>This could be something you read about in the news, or something you did  [MASK]  the weekend.</pre>

**Question**

<pre>Choose the most appropriate answer from the supplied options.</pre>

**Criteria (choice)**

<pre>1. over
2. with
3. from
4. for</pre>

**Gold target (weight)**

<pre>1. over — 1</pre>

</details>
