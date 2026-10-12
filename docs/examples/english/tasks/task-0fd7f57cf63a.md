# nlgraph

Classification · `tasksource/nlgraph`

<details><summary>Example 1</summary>

published Jev · train · nlgraph-a3a6891c80:train:0

**State**

<pre>In an undirected graph, (i,j) means that node i and node j are connected with an undirected edge.
The nodes are numbered from 0 to 24, and the edges are: (0,1) (9,6) (12,20) (13,17) (23,4) (2,0) (1,6) (3,0) (15,19) (14,5) (16,12) (18,21) (2,7) (9,2) (23,13) (17,24) (2,11) (0,4) (4,3) (11,23) (10,23) (5,9) (8,23) (21,12) (16,22) (17,15) (10,12)
Q: Is there a cycle in this graph?
A:</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>2. True — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · nlgraph-a3a6891c80:train:239

**State**

<pre>Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.
Graph: (0,4) (0,18) (0,17) (0,10) (1,7) (1,24) (1,27) (2,4) (2,8) (2,13) (2,19) (2,27) (3,16) (3,28) (3,9) (4,18) (4,6) (4,10) (4,11) (4,27) (6,20) (6,8) (6,7) (6,27) (7,8) (7,18) (7,17) (7,23) (7,24) (7,10) (7,13) (7,19) (7,11) (7,14) (8,17) (8,22) (8,24) (8,10) (8,11) (10,18) (10,17) (10,12) (10,11) (10,27) (11,17) (11,12) (12,20) (12,18) (12,17) (12,14) (13,20) (13,18) (13,24) (13,27) (13,26) (14,18) (15,28) (15,25) (16,25) (17,20) (17,18) (17,26) (18,22) (18,23) (18,19) (18,27) (19,23) (19,27) (19,26) (20,24) (22,23) (22,24) (22,26) (26,27)
Q: Is there a path between node 5 and node 15?
A:</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>1. False — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · nlgraph-a3a6891c80:train:733

**State**

<pre>Determine if there is a path between two nodes in the graph. Note that (i,j) means that node i and node j are connected with an undirected edge.
Graph: (0,18) (1,20) (2,11) (2,23) (2,14) (2,8) (3,17) (3,14) (3,16) (4,7) (4,16) (4,8) (4,24) (5,6) (5,10) (5,20) (6,20) (7,12) (7,16) (7,8) (7,24) (8,12) (8,23) (8,21) (9,18) (10,18) (10,20) (11,17) (11,19) (11,24) (12,22) (12,17) (12,14) (13,24) (14,23) (14,24) (15,17) (15,19) (15,24) (16,22) (16,19) (17,19) (17,24) (19,24) (23,24)
Q: Is there a path between node 22 and node 15?
A:</pre>

**Question**

<pre>Choose the criterion that best describes the state.</pre>

**Criteria (choice)**

<pre>1. False
2. True</pre>

**Gold target (weight)**

<pre>2. True — 1</pre>

</details>
