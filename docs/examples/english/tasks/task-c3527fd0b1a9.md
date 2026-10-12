# conll2003/ner_tags

TokenClassification · `tomaarsen/conll2003`

<details><summary>Example 1</summary>

published Jev · train · conll2003-ner-tags-be686b5302:train:0:token-0

**State**

<pre>Sentence: EU rejects German call to boycott British lamb .
Target token at position 0: EU
Marked sentence: [TARGET: EU] rejects German call to boycott British lamb .</pre>

**Question**

<pre>Choose the criterion that best labels the target token.</pre>

**Criteria (choice)**

<pre>1. outside any named entity
2. beginning of a person entity
3. inside a person entity
4. beginning of an organization entity
5. inside an organization entity
6. beginning of a location entity
7. inside a location entity
8. beginning of a miscellaneous entity
9. inside a miscellaneous entity</pre>

**Gold target (weight)**

<pre>4. beginning of an organization entity — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · conll2003-ner-tags-be686b5302:train:0:token-4

**State**

<pre>Sentence: EU rejects German call to boycott British lamb .
Target token at position 4: to
Marked sentence: EU rejects German call [TARGET: to] boycott British lamb .</pre>

**Question**

<pre>Choose the criterion that best labels the target token.</pre>

**Criteria (choice)**

<pre>1. outside any named entity
2. beginning of a person entity
3. inside a person entity
4. beginning of an organization entity
5. inside an organization entity
6. beginning of a location entity
7. inside a location entity
8. beginning of a miscellaneous entity
9. inside a miscellaneous entity</pre>

**Gold target (weight)**

<pre>1. outside any named entity — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · conll2003-ner-tags-be686b5302:train:0:token-4:choice-criteria-permutation

**State**

<pre>Sentence: EU rejects German call to boycott British lamb .
Target token at position 4: to
Marked sentence: EU rejects German call [TARGET: to] boycott British lamb .</pre>

**Question**

<pre>Choose the criterion that best labels the target token.</pre>

**Criteria (choice)**

<pre>1. outside any named entity
2. inside an organization entity
3. inside a location entity
4. inside a person entity
5. beginning of a person entity
6. beginning of a miscellaneous entity
7. beginning of an organization entity
8. inside a miscellaneous entity
9. beginning of a location entity</pre>

**Gold target (weight)**

<pre>1. outside any named entity — 1</pre>

</details>
