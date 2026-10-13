# cebab/rating

Classification · `EleutherAI/CEBaB`

<details><summary>Example 1</summary>

published Jev · train · cebab:train:train:3201:cebab-rating-aa208c51c2

**State**

<pre>Food was good. The noise level was very quiet. Hostess was rude and mean. Would not go back. Not even for the food.</pre>

**Question**

<pre>How would readers rate this restaurant review, from 1 star (worst) to 5 stars (best)? Predict the median rating.</pre>

**Criteria (score)**

<pre>1. 1 star
2. 2 stars
3. 3 stars
4. 4 stars
5. 5 stars</pre>

**Gold target (weight)**

<pre>1. 1 star — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · cebab:train:train:11687:cebab-rating-aa208c51c2

**State**

<pre>Nothing to write home about.</pre>

**Question**

<pre>How would readers rate this restaurant review, from 1 star (worst) to 5 stars (best)? Predict the median rating.</pre>

**Criteria (score)**

<pre>1. 1 star
2. 2 stars
3. 3 stars
4. 4 stars
5. 5 stars</pre>

**Gold target (weight)**

<pre>2. 2 stars — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · cebab:train:train:906:cebab-rating-aa208c51c2

**State**

<pre>Hello
I expected more from this restaurant. The pasta was excellent, but the service was not good and the entree arrived over 30 minutes after the pasta.</pre>

**Question**

<pre>How would readers rate this restaurant review, from 1 star (worst) to 5 stars (best)? Predict the median rating.</pre>

**Criteria (score)**

<pre>1. 1 star
2. 2 stars
3. 3 stars
4. 4 stars
5. 5 stars</pre>

**Gold target (weight)**

<pre>3. 3 stars — 1</pre>

</details>
