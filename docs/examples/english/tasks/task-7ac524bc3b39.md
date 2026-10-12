# HelpSteer2/coherence

Classification · `nvidia/HelpSteer2`

<details><summary>Example 1</summary>

published Jev · train · HelpSteer2-coherence-06fbf6346d:train:pack-5da8f0a2d8f1:label-B

**State**

<pre>Item A:
text_A: User: Assume you are an air traffic controller. How would you deal with an emergency fuel situation?

Assistant: Great question! First, I’d need to determine the severity of the situation, by checking on the availability of alternative airports, assessing the weather, and evaluating the fuel requirements for the remainder of the flight. Once I had a better sense of the circumstances, I’d communicate with the pilot to determine if they could adjust their flight plan in order to reduce fuel usage. I’d also notify air traffic control of the emergency, and request any available resources to assist with the situation. Finally, I’d coordinate with the cabin crew to make sure everyone is safe, and prepare for any emergency procedures that might be necessary.

User: Write a communication to me as if I am the pilot of the aircraft. My callsign is speedbird 16. You are based at London Area Control Centre.
text_B: Hello Speedbird 1

Item B:
text_A: What is the difference between Black&#x27;s BBQ and Terry Black&#x27;s BBQ?
text_B: I don&#x27;t know. Can you tell me more about them?

Item C:
text_A: sdf
text_B: I&#x27;m sorry, but I don&#x27;t understand what you mean by &quot;sdf&quot;. If you have a question or need assistance with something, I&#x27;d be happy to help you, just let me know what it is that you need.</pre>

**Question**

<pre>Each item answers: &quot;How would you rate the coherence of the response?&quot;
Choose the criterion that best describes Item B.</pre>

**Criteria (choice)**

<pre>1. 0: incoherent
2. 1
3. 2
4. 3
5. 4: perfectly clear</pre>

**Gold target (weight)**

<pre>1. 0: incoherent — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · HelpSteer2-coherence-06fbf6346d:train:pack-5da8f0a2d8f1:same-A-C

**State**

<pre>Item A:
text_A: User: Assume you are an air traffic controller. How would you deal with an emergency fuel situation?

Assistant: Great question! First, I’d need to determine the severity of the situation, by checking on the availability of alternative airports, assessing the weather, and evaluating the fuel requirements for the remainder of the flight. Once I had a better sense of the circumstances, I’d communicate with the pilot to determine if they could adjust their flight plan in order to reduce fuel usage. I’d also notify air traffic control of the emergency, and request any available resources to assist with the situation. Finally, I’d coordinate with the cabin crew to make sure everyone is safe, and prepare for any emergency procedures that might be necessary.

User: Write a communication to me as if I am the pilot of the aircraft. My callsign is speedbird 16. You are based at London Area Control Centre.
text_B: Hello Speedbird 1

Item B:
text_A: What is the difference between Black&#x27;s BBQ and Terry Black&#x27;s BBQ?
text_B: I don&#x27;t know. Can you tell me more about them?

Item C:
text_A: sdf
text_B: I&#x27;m sorry, but I don&#x27;t understand what you mean by &quot;sdf&quot;. If you have a question or need assistance with something, I&#x27;d be happy to help you, just let me know what it is that you need.</pre>

**Question**

<pre>Each item answers: &quot;How would you rate the coherence of the response?&quot;
Do Item A and Item C have the same label? Possible labels: &quot;0: incoherent&quot;, &quot;1&quot;, &quot;2&quot;, &quot;3&quot;, &quot;4: perfectly clear&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · HelpSteer2-coherence-06fbf6346d:train:pack-5da8f0a2d8f1:exists-4

**State**

<pre>Item A:
text_A: User: Assume you are an air traffic controller. How would you deal with an emergency fuel situation?

Assistant: Great question! First, I’d need to determine the severity of the situation, by checking on the availability of alternative airports, assessing the weather, and evaluating the fuel requirements for the remainder of the flight. Once I had a better sense of the circumstances, I’d communicate with the pilot to determine if they could adjust their flight plan in order to reduce fuel usage. I’d also notify air traffic control of the emergency, and request any available resources to assist with the situation. Finally, I’d coordinate with the cabin crew to make sure everyone is safe, and prepare for any emergency procedures that might be necessary.

User: Write a communication to me as if I am the pilot of the aircraft. My callsign is speedbird 16. You are based at London Area Control Centre.
text_B: Hello Speedbird 1

Item B:
text_A: What is the difference between Black&#x27;s BBQ and Terry Black&#x27;s BBQ?
text_B: I don&#x27;t know. Can you tell me more about them?

Item C:
text_A: sdf
text_B: I&#x27;m sorry, but I don&#x27;t understand what you mean by &quot;sdf&quot;. If you have a question or need assistance with something, I&#x27;d be happy to help you, just let me know what it is that you need.</pre>

**Question**

<pre>Each item answers: &quot;How would you rate the coherence of the response?&quot;
Does at least one item have the label &quot;4: perfectly clear&quot;? Possible labels: &quot;0: incoherent&quot;, &quot;1&quot;, &quot;2&quot;, &quot;3&quot;, &quot;4: perfectly clear&quot;.</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 1</pre>

</details>
