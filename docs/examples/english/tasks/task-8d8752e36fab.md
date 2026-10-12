# mutual

MultipleChoice · `tasksource/mutual`

<details><summary>Example 1</summary>

published Jev · train · mutual-410e5f4f25:train:7

**State**

<pre>f : have you heard there &#x27;s a car accident just in the opposite of the street ? m : no . i have just returned from london . f : 2 people lost their lives at once . m : the road condition needs improving for long . f : it was late and there were few cars . m : that may make a driver careless than during rush hour . f : right . and it said that the driver drank too much .</pre>

**Question**

<pre>Choose the criterion that best answers the question.</pre>

**Criteria (choice)**

<pre>1. m : i see . so the car was overloaded , and three of the passengers lost their lives in the accident .
2. m : poor driver . the accident was not his fault , just that there were too many cars on the street .
3. m : how could the driver took alcohol before driving ? he should be blame for the accident .
4. m : the driver was drunk ? he should take the consequences . fortunately , nobody was hurt .</pre>

**Gold target (weight)**

<pre>3. m : how could the driver took alcohol before driving ? he should be blame for the accident . — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · mutual-410e5f4f25:train:7:choice-instruction-paraphrase

**State**

<pre>f : have you heard there &#x27;s a car accident just in the opposite of the street ? m : no . i have just returned from london . f : 2 people lost their lives at once . m : the road condition needs improving for long . f : it was late and there were few cars . m : that may make a driver careless than during rush hour . f : right . and it said that the driver drank too much .</pre>

**Question**

<pre>Choose the most appropriate answer from the supplied options.</pre>

**Criteria (choice)**

<pre>1. m : i see . so the car was overloaded , and three of the passengers lost their lives in the accident .
2. m : poor driver . the accident was not his fault , just that there were too many cars on the street .
3. m : how could the driver took alcohol before driving ? he should be blame for the accident .
4. m : the driver was drunk ? he should take the consequences . fortunately , nobody was hurt .</pre>

**Gold target (weight)**

<pre>3. m : how could the driver took alcohol before driving ? he should be blame for the accident . — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · mutual-410e5f4f25:train:3022

**State**

<pre>f : good afternoon , what can i do for you ? m : yes , please . i would like to know something about the driving courses . f : well , we have short full time courses during the summer . are you interested in them ? m : no , i am free only at weekends . f : then there are weekend courses . the course starts at 8:00 every saturday and sunday morning . m : sounds fine . what about the coaches ?</pre>

**Question**

<pre>Choose the most appropriate answer from the supplied options.</pre>

**Criteria (choice)**

<pre>1. f : since you have chose the weekend courses , i have to tell you the courses will start at 9:00 everyday .
2. f : if you want to take full-time winter courses since you are free during weekends as well as winter .
3. f : do you mean that you would like to take full-time summer courses to learn how to repair cars ?
4. f : good choice . weekend courses are special made for people who only have time on weekends , just like you .</pre>

**Gold target (weight)**

<pre>4. f : good choice . weekend courses are special made for people who only have time on weekends , just like you . — 1</pre>

</details>
