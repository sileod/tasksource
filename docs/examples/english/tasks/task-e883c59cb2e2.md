# github-issue-similarity

Classification · `WhereIsAI/github-issue-similarity`

<details><summary>Example 1</summary>

published Jev · train · github-issue-similarity-9758040744:train:132

**State**

<pre>First text:
See
https://groups.google.com/d/msg/elasticsearch/OZsc5ofNhag/9JfwSy-CEMwJ where Shay says &quot;...with the design of the facets, if shouldn&#x27;t be hard to implement...&quot;

Second text:
This might sound entirely trivial, but this would be a real boon for consistency in our aggregation parsing code. doc_count is currently a property of each bucket in a bucket agg, which makes it an exception when walking the aggregation tree.
If we could get a dedicated doc_count metric agg we could handle it the same way we handle every other metric agg, even if its only purpose is to duplicate the doc_count property of the bucket.</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. dissimilar issues
2. similar issues</pre>

**Gold target (weight)**

<pre>1. dissimilar issues — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · github-issue-similarity-9758040744:train:132:choice-criteria-permutation

**State**

<pre>First text:
See
https://groups.google.com/d/msg/elasticsearch/OZsc5ofNhag/9JfwSy-CEMwJ where Shay says &quot;...with the design of the facets, if shouldn&#x27;t be hard to implement...&quot;

Second text:
This might sound entirely trivial, but this would be a real boon for consistency in our aggregation parsing code. doc_count is currently a property of each bucket in a bucket agg, which makes it an exception when walking the aggregation tree.
If we could get a dedicated doc_count metric agg we could handle it the same way we handle every other metric agg, even if its only purpose is to duplicate the doc_count property of the bucket.</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. similar issues
2. dissimilar issues</pre>

**Gold target (weight)**

<pre>2. dissimilar issues — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · github-issue-similarity-9758040744:train:14022

**State**

<pre>Passage A:
Expected Behavior
Use the session.setProxy function of a window to set a user:pass authenticated proxy. Load URL in that window. Expect that app.login event is triggered and request is made.
Actual behavior
Set session proxy to a user:pass authenticated proxy, loadURL requests fail with status &quot;failed&quot; in dev tools. App.login event never called.
To Reproduce
Create a window, use the session setProxy function to set a user:pass authenticated proxy. Then try to load ANY url in that window.
win.webContents.session.setProxy({
          proxyRules: `http://user:pass@ip:port`
        }, () =&gt; {});
 }
//Requests fail status &quot;failed&quot;
win.loadURL(&#x27;https://whatismyipaddress.com&#x27;);
//This is NOT reached
app.on(&#x27;login&#x27;, function (event, request, authInfo, callback) {
  if (authInfo.isProxy) {
    log.info(`HI ${JSON.stringify(authInfo)}`)
    // callback(&#x27;username&#x27;, &#x27;password&#x27;);
  }
})
Working code with IP authenticated proxies
win.webContents.session.setProxy({
          proxyRules: `http://ip:port`
        }, () =&gt; {});
 }
//Requests succeed
win.loadURL(&#x27;https://whatismyipaddress.com&#x27;);
//This is reached
app.on(&#x27;login&#x27;, function (event, request, authInfo, callback) {
  if (authInfo.isProxy) {
    log.info(`HI ${JSON.stringify(authInfo)}`)
    // callback(&#x27;username&#x27;, &#x27;password&#x27;);
  }
})
This code is working for me with IP authenticated proxies. App.login is triggered and the request completes as expected. Just user:pass proxies that are the issue.
When opening chrome dev tools I can see the requests being attempted and returning status &quot;failed&quot;.
**Screenshots**
If applicable, add screenshots to help explain your problem.
**Additional Information**
Add any other context about the problem here.

Passage B:
Hello,
I&#x27;m hoping for some general guidance... it seems when using electron notifications (either with the html5 api in from within renderer process or the notification module inside the main process) each notification fires off three times, resulting in three identical notifications and an annoyed user. My best guess is this is due to electron&#x27;s multi-process architecture (I can see three instances of my app running).
Luckily on macOS (if using the html 5 api) the OS is smart enough to recognize the 3 notifications are identical and collapses them into one, not so lucky if the notifications are coming from the main process using the notification module, and likewise on windows from both main and render processes.
I could get into the details of how I&#x27;m setting up notifications, but this doesn&#x27;t seem directly related to notifications rather more to the electron architecture. Anyone have any thoughts or wisdom on how I can work around this?
Electron version: 1.8.4
Operating system: macOS/Windows (Build only)
Expected behavior
Call notification... e.g.:
Render Process
let notify = new Notification(&#x27;Notification Title&#x27;, {
  body: &#x27;notification body&#x27;
});
or
Main Process
let notify = new Notification({
  title: &#x27;Notification Title&#x27;,
  body: &#x27;notification body&#x27;
});
... see one desktop notification.
Actual behavior
Three desktop notifications are fired.</pre>

**Question**

<pre>Choose the most appropriate category for the state.</pre>

**Criteria (choice)**

<pre>1. dissimilar issues
2. similar issues</pre>

**Gold target (weight)**

<pre>1. dissimilar issues — 1</pre>

</details>
