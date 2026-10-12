# acronym_identification

TokenClassification · `amirveyseh/acronym_identification`

<details><summary>Example 1</summary>

canonical preprocessing (not a published Jev row) · train · acronym_identification:canonical:sample:0

**Question**

<pre>Tag acronym short forms and their expanded long forms: B starts a span, I continues it, O is outside.</pre>

**Criteria (label vocabulary)**

<pre>B-long
B-short
I-long
I-short
O</pre>

**Tokens / gold tags**

<pre>We → O
use → O
the → O
wide → B-long
residual → I-long
network → I-long
( → O
WRN → B-short
) → O
architecture → O
to → O
evaluate → O
the → O
effect → O
of → O
sparsity → O
on → O
classification → O
accuracy → O
using → O
Sparseout → O
. → O</pre>

</details>

<details><summary>Example 2</summary>

canonical preprocessing (not a published Jev row) · train · acronym_identification:canonical:sample:1

**Question**

<pre>Tag acronym short forms and their expanded long forms: B starts a span, I continues it, O is outside.</pre>

**Criteria (label vocabulary)**

<pre>B-long
B-short
I-long
I-short
O</pre>

**Tokens / gold tags**

<pre>Combining → O
the → O
above → O
steps → O
, → O
we → O
can → O
now → O
formally → O
introduce → O
the → O
proposed → O
distributed → O
algorithm → O
for → O
the → O
DL → O
problems → O
, → O
as → O
described → O
in → O
Algorithm → O
, → O
and → O
termed → O
DL → B-short
( → O
Decentralized → O
Dictionary → O
Learning → O
over → O
Dynamic → O
Digraphs → O
) → O
Algorithm → O
. → O</pre>

</details>

<details><summary>Example 3</summary>

canonical preprocessing (not a published Jev row) · train · acronym_identification:canonical:sample:2

**Question**

<pre>Tag acronym short forms and their expanded long forms: B starts a span, I continues it, O is outside.</pre>

**Criteria (label vocabulary)**

<pre>B-long
B-short
I-long
I-short
O</pre>

**Tokens / gold tags**

<pre>Recently → O
, → O
a → O
new → O
benchmark → O
has → O
become → O
available → O
to → O
address → O
the → O
specific → O
needs → O
of → O
ND → B-short
detection → O
: → O
the → O
Mir → B-long
- → I-long
Flickr → I-long
Near → I-long
Duplicate → I-long
( → O
MFND → B-short
) → O
dataset → O
, → O
based → O
on → O
the → O
pre → O
- → O
existing → O
MIR → B-short
- → O
Flickr → O
collection → O
. → O</pre>

</details>
