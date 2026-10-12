# blog_authorship_corpus/age

Classification · `tasksource/blog_authorship_corpus`

<details><summary>Example 1</summary>

published Jev · train · blog-authorship-corpus-age-8880d04019:train:pack-b0848e7c5c3e:label-A

**State**

<pre>Item A:
           Seems like everyone gat ride problems....I ain ga b able 2 go unless danny n jess come (that wul b my ride back up 2 bayview cuz i sure as hell ain walkin there!). I think we jus gone n planned all this stuff on a really messed up weekend dread...but I got some good news (along with bad news). I got my phone yesterday and after all the trouble I went through to carry it to batelco n stuff, they transferred it (so they say) but it ain workin yet (xcept n e time i get a voice message it comes up on the phone-das sum weird crap!) So yea, hopefully we ga be able to get to go out 2nite, but I kinda gat the feeling that if we do....it ga b really jacked up dred! -L8a!          

Item B:
           &#x27;Mom! I think Emily has PMS. I think she has her period because she has PMS.&#x27; - Mitchel &#x27;UGH STAY OUT OF MY LIFE!&#x27; - Me  &#x27;Miraculous recovery. Yesterday on crutches today walking perfectly. How amazing.&#x27; - Mr. Flangan &#x27;Yeah, how miraculous.&#x27; - Me  &#x27;We&#x27;ll watch the rest of the video after we correct the video tomorrow.&#x27; - Mr. Flanagan  &#x27;Let me see how I do putting your glasses on today.&#x27; - Lori *I laugh as she completely misses my ears.*         

Item C:
        urlLink     What internet acronym are you?     brought to you by  urlLink Quizilla       </pre>

**Question**

<pre>Each item answers: &quot;What is the blogger&#x27;s age group?&quot;
Choose the criterion that best describes Item A.</pre>

**Criteria (choice)**

<pre>1. 13-17
2. 23-27
3. 33-48</pre>

**Gold target (weight)**

<pre>1. 13-17 — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · blog-authorship-corpus-age-8880d04019:train:pack-b0848e7c5c3e:same-A-B

**State**

<pre>Item A:
           Seems like everyone gat ride problems....I ain ga b able 2 go unless danny n jess come (that wul b my ride back up 2 bayview cuz i sure as hell ain walkin there!). I think we jus gone n planned all this stuff on a really messed up weekend dread...but I got some good news (along with bad news). I got my phone yesterday and after all the trouble I went through to carry it to batelco n stuff, they transferred it (so they say) but it ain workin yet (xcept n e time i get a voice message it comes up on the phone-das sum weird crap!) So yea, hopefully we ga be able to get to go out 2nite, but I kinda gat the feeling that if we do....it ga b really jacked up dred! -L8a!          

Item B:
           &#x27;Mom! I think Emily has PMS. I think she has her period because she has PMS.&#x27; - Mitchel &#x27;UGH STAY OUT OF MY LIFE!&#x27; - Me  &#x27;Miraculous recovery. Yesterday on crutches today walking perfectly. How amazing.&#x27; - Mr. Flangan &#x27;Yeah, how miraculous.&#x27; - Me  &#x27;We&#x27;ll watch the rest of the video after we correct the video tomorrow.&#x27; - Mr. Flanagan  &#x27;Let me see how I do putting your glasses on today.&#x27; - Lori *I laugh as she completely misses my ears.*         

Item C:
        urlLink     What internet acronym are you?     brought to you by  urlLink Quizilla       </pre>

**Question**

<pre>Each item answers: &quot;What is the blogger&#x27;s age group?&quot;
Do Item A and Item B have the same label? Possible labels: &quot;13-17&quot;, &quot;23-27&quot;, &quot;33-48&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · blog-authorship-corpus-age-8880d04019:train:pack-b0848e7c5c3e:all-same

**State**

<pre>Item A:
           Seems like everyone gat ride problems....I ain ga b able 2 go unless danny n jess come (that wul b my ride back up 2 bayview cuz i sure as hell ain walkin there!). I think we jus gone n planned all this stuff on a really messed up weekend dread...but I got some good news (along with bad news). I got my phone yesterday and after all the trouble I went through to carry it to batelco n stuff, they transferred it (so they say) but it ain workin yet (xcept n e time i get a voice message it comes up on the phone-das sum weird crap!) So yea, hopefully we ga be able to get to go out 2nite, but I kinda gat the feeling that if we do....it ga b really jacked up dred! -L8a!          

Item B:
           &#x27;Mom! I think Emily has PMS. I think she has her period because she has PMS.&#x27; - Mitchel &#x27;UGH STAY OUT OF MY LIFE!&#x27; - Me  &#x27;Miraculous recovery. Yesterday on crutches today walking perfectly. How amazing.&#x27; - Mr. Flangan &#x27;Yeah, how miraculous.&#x27; - Me  &#x27;We&#x27;ll watch the rest of the video after we correct the video tomorrow.&#x27; - Mr. Flanagan  &#x27;Let me see how I do putting your glasses on today.&#x27; - Lori *I laugh as she completely misses my ears.*         

Item C:
        urlLink     What internet acronym are you?     brought to you by  urlLink Quizilla       </pre>

**Question**

<pre>Each item answers: &quot;What is the blogger&#x27;s age group?&quot;
Do all items have the same label? Possible labels: &quot;13-17&quot;, &quot;23-27&quot;, &quot;33-48&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
