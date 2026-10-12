# lifecycle-entailment

Classification · `tasksource/lifecycle-entailment`

<details><summary>Example 1</summary>

published Jev · train · lifecycle-entailment-2e88a896f6:train:106

**State**

<pre>text_A: ::stage egg:: It all starts when a female butterfly lays her eggs, usually on leaves or stems of plants. Inside these tiny eggs, caterpillars grow. Depending on the species, the eggs can vary in shape and texture - they can be round, oval or cylindrical, and smooth, bumpy or wrinkled. The time it takes for the eggs to hatch can also vary - in some species, they will hatch within a few weeks and in others they will only hatch once the weather is warm enough. ::stage caterpillar:: Once ready, the caterpillar leaves its egg home and enters the big outside world! And these little critters have one serious appetite - the caterpillars actually eat their way out of the egg and immediately start chomping on the leaves of the host plant. During this stage, they shed their skin four or five times - as the caterpillar grows, its skin becomes too tight and splits open, revealing a new, larger skin underneath. A fully grown caterpillar can be over 100 times larger than when it emerged from its egg. Wow! ::stage pupa:: Once fully grown, the caterpillar forms itself into a &#x27;pupa&#x27; (or chrysalis) - a kind of vessel in which the caterpillar changes into a butterfly. The caterpillars usually do this on twigs or safe, hidden areas around the host plant. The &#x27;pupa&#x27; stage may last a few weeks to several months depending on the species. During this time, a hardened case forms around the pupa to protect it from predators and extreme weather conditions. And inside, the tissue, limbs and organs of the caterpillar transform. The result? A wonderful winged butterfly! ::stage adult:: Once the butterfly is ready to emerge, the case around the pupa splits open. But it&#x27;s not time for the adult butterfly to take off just yet, as the wings are at first wet, soft and wrinkled against its body. The butterfly waits for its wings to dry, and pumps a liquid called hemolymph into them so that they become big and strong. Once fit for flight, this brilliant bug then takes to the air in search for flowers to feed on and for other butterflies to mate with. And that&#x27;s the cycle complete - and ready to start all over again! 
text_B: Twigs do a butterfly pupa go through metamorphosis.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · lifecycle-entailment-2e88a896f6:train:106:choice-paired-text-format

**State**

<pre>Passage A:
::stage egg:: It all starts when a female butterfly lays her eggs, usually on leaves or stems of plants. Inside these tiny eggs, caterpillars grow. Depending on the species, the eggs can vary in shape and texture - they can be round, oval or cylindrical, and smooth, bumpy or wrinkled. The time it takes for the eggs to hatch can also vary - in some species, they will hatch within a few weeks and in others they will only hatch once the weather is warm enough. ::stage caterpillar:: Once ready, the caterpillar leaves its egg home and enters the big outside world! And these little critters have one serious appetite - the caterpillars actually eat their way out of the egg and immediately start chomping on the leaves of the host plant. During this stage, they shed their skin four or five times - as the caterpillar grows, its skin becomes too tight and splits open, revealing a new, larger skin underneath. A fully grown caterpillar can be over 100 times larger than when it emerged from its egg. Wow! ::stage pupa:: Once fully grown, the caterpillar forms itself into a &#x27;pupa&#x27; (or chrysalis) - a kind of vessel in which the caterpillar changes into a butterfly. The caterpillars usually do this on twigs or safe, hidden areas around the host plant. The &#x27;pupa&#x27; stage may last a few weeks to several months depending on the species. During this time, a hardened case forms around the pupa to protect it from predators and extreme weather conditions. And inside, the tissue, limbs and organs of the caterpillar transform. The result? A wonderful winged butterfly! ::stage adult:: Once the butterfly is ready to emerge, the case around the pupa splits open. But it&#x27;s not time for the adult butterfly to take off just yet, as the wings are at first wet, soft and wrinkled against its body. The butterfly waits for its wings to dry, and pumps a liquid called hemolymph into them so that they become big and strong. Once fit for flight, this brilliant bug then takes to the air in search for flowers to feed on and for other butterflies to mate with. And that&#x27;s the cycle complete - and ready to start all over again! 

Passage B:
Twigs do a butterfly pupa go through metamorphosis.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · lifecycle-entailment-2e88a896f6:train:1642

**State**

<pre>text_A: ::stage Egg:: A male and a female dragonfly will mate while they are flying in the air. After two dragonflies mate, the female dragonfly will lay her eggs on a plant in the water, or if she can&#x27;t find a suitable plant she will just drop them into the water. ::stage Nymph:: Once the dragonfly eggs hatch, the life cycle of a dragonfly larva begins as a nymph. A nymph looks like a little alien creature. Nymph hasn&#x27;t grown its wings yet and has what looks like a crusty hump hanging onto its back. Dragonfly nymphs live in the water while they grow and develop into dragonflies. This portion of the dragonfly life cycle can take up to four years to complete, and if the nymph cycle is completed in the beginning of the wintertime, it will remain in the water until spring when it is warm enough to come out. Dragonfly nymphs live in ponds or marshy areas because the waters are calmer than in a stream or river. Sometimes they can be found in the calmer backwaters of rivers, too. Dragonfly nymphs may eat smaller dragonfly nymphs as they develop. ::stage Adult:: Once the nymph is fully grown, and the weather is right, it will complete the metamorphosis into a dragonfly by crawling out of the water up the stem of a plant. The nymph will shed its skin onto the stem of the plant and will then be a young dragonfly. The skin that the nymph left behind is called the exuvia and you can find the exuvia still stuck to the stem for a long time after the dragonfly has left it. Once the dragonfly leaves the exuvia it is a full grown dragonfly. The dragonfly will hunt for food and begin to look for a mate. Once the dragonfly finds a mate, the female will find a body of calm water that will be a good place to lay her eggs, and the life cycle of the dragonfly begins all over again. Adult dragonflies only live about two months. 
text_B: During adult stage do dragonflies begin to look for a mate.</pre>

**Question**

<pre>Does text_A entail text_B?</pre>

**Criteria (choice)**

<pre>1. entailment
2. not_entailment</pre>

**Gold target (weight)**

<pre>1. entailment — 1</pre>

</details>
