# clinc_oos/plus

Classification · `clinc/clinc_oos`

<details><summary>Example 1</summary>

published Jev · train · clinc-oos-plus-1b9b3d1a5a:train:28

**State**

<pre>translate hello to english</pre>

**Question**

<pre>Which of the supplied criteria best matches the state?</pre>

**Criteria (choice)**

<pre>1. restaurant_reviews
2. nutrition_info
3. account_blocked
4. oil_change_how
5. time
6. weather
7. redeem_rewards
8. interest_rate
9. gas_type
10. accept_reservations
11. smart_home
12. user_name
13. report_lost_card
14. repeat
15. whisper_mode
16. what_are_your_hobbies
17. order
18. jump_start
19. schedule_meeting
20. meeting_schedule
21. freeze_account
22. what_song
23. meaning_of_life
24. restaurant_reservation
25. traffic
26. make_call
27. text
28. bill_balance
29. improve_credit_score
30. change_language
31. no
32. measurement_conversion
33. timer
34. flip_coin
35. do_you_have_pets
36. balance
37. tell_joke
38. last_maintenance
39. exchange_rate
40. uber
41. car_rental
42. credit_limit
43. oos
44. shopping_list
45. expiration_date
46. routing
47. meal_suggestion
48. tire_change
49. todo_list
50. card_declined
51. rewards_balance
52. change_accent
53. vaccines
54. reminder_update
55. food_last
56. change_ai_name
57. bill_due
58. who_do_you_work_for
59. share_location
60. international_visa
61. calendar
62. translate
63. carry_on
64. book_flight
65. insurance_change
66. todo_list_update
67. timezone
68. cancel_reservation
69. transactions
70. credit_score
71. report_fraud
72. spending_history
73. directions
74. spelling
75. insurance
76. what_is_your_name
77. reminder
78. where_are_you_from
79. distance
80. payday
81. flight_status
82. find_phone
83. greeting
84. alarm
85. order_status
86. confirm_reservation
87. cook_time
88. damaged_card
89. reset_settings
90. pin_change
91. replacement_card_duration
92. new_card
93. roll_dice
94. income
95. taxes
96. date
97. who_made_you
98. pto_request
99. tire_pressure
100. how_old_are_you
101. rollover_401k
102. pto_request_status
103. how_busy
104. application_status
105. recipe
106. calendar_update
107. play_music
108. yes
109. direct_deposit
110. credit_limit_change
111. gas
112. pay_bill
113. ingredients_list
114. lost_luggage
115. goodbye
116. what_can_i_ask_you
117. book_hotel
118. are_you_a_bot
119. next_song
120. change_speed
121. plug_type
122. maybe
123. w2
124. oil_change_when
125. thank_you
126. shopping_list_update
127. pto_balance
128. order_checks
129. travel_alert
130. fun_fact
131. sync_device
132. schedule_maintenance
133. apr
134. transfer
135. ingredient_substitution
136. calories
137. current_location
138. international_fees
139. calculator
140. definition
141. next_holiday
142. update_playlist
143. mpg
144. min_payment
145. change_user_name
146. restaurant_suggestion
147. travel_notification
148. cancel
149. pto_used
150. travel_suggestion
151. change_volume</pre>

**Gold target (weight)**

<pre>62. translate — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · clinc-oos-plus-1b9b3d1a5a:train:28:noul-label-verification

**State**

<pre>translate hello to english</pre>

**Question**

<pre>Is &quot;order&quot; the correct label for this example?</pre>

**Decision**

<pre>Boolean (noul)</pre>

**Gold target**

<pre>Probability of true: 0</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · clinc-oos-plus-1b9b3d1a5a:train:12544

**State**

<pre>find me a roundtrip flight from dallas to los angeles from march 2 to march 10</pre>

**Question**

<pre>Select the label that best applies to the state.</pre>

**Criteria (choice)**

<pre>1. restaurant_reviews
2. nutrition_info
3. account_blocked
4. oil_change_how
5. time
6. weather
7. redeem_rewards
8. interest_rate
9. gas_type
10. accept_reservations
11. smart_home
12. user_name
13. report_lost_card
14. repeat
15. whisper_mode
16. what_are_your_hobbies
17. order
18. jump_start
19. schedule_meeting
20. meeting_schedule
21. freeze_account
22. what_song
23. meaning_of_life
24. restaurant_reservation
25. traffic
26. make_call
27. text
28. bill_balance
29. improve_credit_score
30. change_language
31. no
32. measurement_conversion
33. timer
34. flip_coin
35. do_you_have_pets
36. balance
37. tell_joke
38. last_maintenance
39. exchange_rate
40. uber
41. car_rental
42. credit_limit
43. oos
44. shopping_list
45. expiration_date
46. routing
47. meal_suggestion
48. tire_change
49. todo_list
50. card_declined
51. rewards_balance
52. change_accent
53. vaccines
54. reminder_update
55. food_last
56. change_ai_name
57. bill_due
58. who_do_you_work_for
59. share_location
60. international_visa
61. calendar
62. translate
63. carry_on
64. book_flight
65. insurance_change
66. todo_list_update
67. timezone
68. cancel_reservation
69. transactions
70. credit_score
71. report_fraud
72. spending_history
73. directions
74. spelling
75. insurance
76. what_is_your_name
77. reminder
78. where_are_you_from
79. distance
80. payday
81. flight_status
82. find_phone
83. greeting
84. alarm
85. order_status
86. confirm_reservation
87. cook_time
88. damaged_card
89. reset_settings
90. pin_change
91. replacement_card_duration
92. new_card
93. roll_dice
94. income
95. taxes
96. date
97. who_made_you
98. pto_request
99. tire_pressure
100. how_old_are_you
101. rollover_401k
102. pto_request_status
103. how_busy
104. application_status
105. recipe
106. calendar_update
107. play_music
108. yes
109. direct_deposit
110. credit_limit_change
111. gas
112. pay_bill
113. ingredients_list
114. lost_luggage
115. goodbye
116. what_can_i_ask_you
117. book_hotel
118. are_you_a_bot
119. next_song
120. change_speed
121. plug_type
122. maybe
123. w2
124. oil_change_when
125. thank_you
126. shopping_list_update
127. pto_balance
128. order_checks
129. travel_alert
130. fun_fact
131. sync_device
132. schedule_maintenance
133. apr
134. transfer
135. ingredient_substitution
136. calories
137. current_location
138. international_fees
139. calculator
140. definition
141. next_holiday
142. update_playlist
143. mpg
144. min_payment
145. change_user_name
146. restaurant_suggestion
147. travel_notification
148. cancel
149. pto_used
150. travel_suggestion
151. change_volume</pre>

**Gold target (weight)**

<pre>64. book_flight — 1</pre>

</details>
