# weblinx/dom-element

MultipleChoice · `McGill-NLP/WebLINX`

<details><summary>Example 1</summary>

published Jev · train · weblinx:train:weblinx:frvoxei:26:weblinx-dom-element-815194ebdc

**State**

<pre>User dialogue:
[-00:09] Hi [00:03] Open brandmark. [00:19] Create a logo for an active wear brand named “Flex Fit”. [01:13] Add the below slogan: 
	
	&quot;Master Your Vision, Elevate Your Style!&quot; ;
Previous actions:
click(uid=&quot;c22514b8-3126-48ae&quot;) paste(text=&quot;Master Your Vision, Elevate Your Style!&quot;, timestamp=&quot;01:27&quot;) text_input(text=&quot;Master Your Vision, Elevate Your Style!&quot;, uid=&quot;c22514b8-3126-48ae&quot;) click(uid=&quot;65922ce5-06d5-40c7&quot;) click(uid=&quot;6d5f9829-2d49-4eb1&quot;)&lt;/s&gt;&lt;s&gt;[INST]
DOM:
(html(div(div class=&quot;&quot;(div class=&quot;wizard-panel keywords-wizard-panel active&quot;(div class=&quot;wizard-card&quot;(div class=&quot;wizard-card-content&quot;(div(h2 data-webtasks-id=&quot;8388af11-580a-4f37&quot;BRAND KEYWORDS)(pEnter a few words that describe your brand.)(p(span data-webtasks-id=&quot;f88f49f2-bbd7-41c0&quot;Not sure what to use? )(button class=&quot;btn pale-teal&quot;See examples)))))(div class=&quot;wizard-card&quot;(div class=&quot;wizard-card-content&quot;(div(div class=&quot;vue-input-tag-wrapper&quot; data-webtasks-id=&quot;584fe3a7-29a7-4c12&quot;(span class=&quot;input-tag&quot; data-webtasks-id=&quot;c147b1a9-054f-4959&quot;(span data-webtasks-id=&quot;1760e1f2-cc86-434d&quot;RunWithFire)(a class=&quot;remove&quot; data-webtasks-id=&quot;6d5f9829-2d49-4eb1&quot;))(span class=&quot;input-tag&quot; data-webtasks-id=&quot;815c99ad-b305-4961&quot;(a class=&quot;remove&quot; data-webtasks-id=&quot;542eb267-463d-45ea&quot;))(input placeholder=&quot;&quot; type=&quot;text&quot; class=&quot;new-tag&quot; spellcheck=&quot;false&quot; data-webtasks-id=&quot;d08aa426-c133-4169&quot;)))))(div class=&quot;wizard-panel next&quot;)(div class=&quot;wizard-panel next&quot;)(button class=&quot;next-step-btn&quot; data-webtasks-id=&quot;5cd1c446-cdab-4da4&quot;(svg width=&quot;25.805&quot; height=&quot;44.891&quot; viewbox=&quot;0 0 25.805 44.891&quot;(path d=&quot;M8806.227,558.43l8.188-8.189c10.092-10.091,10.926-11.013,10.926-11.013...09-9.016,9-10.912,10.875C8808.4,560.6,8806.227,558.43,8806.227,558.43Z&quot; transform=&quot;translate(-8805.419 -516.783)&quot; fill=&quot;#fff&quot; stroke=&quot;#fff&quot; stroke-miterlimit=&quot;10&quot; stroke-width=&quot;1&quot;))))))
Operation: click</pre>

**Question**

<pre>Which candidate DOM element should receive the specified operation?</pre>

**Criteria (choice)**

<pre>1. (uid = 5cd1c446-cdab-4da4) [[tag]] button [[xpath]] /html/body.../div/button [[bbox]] x=1101 y=37 width=132 height=575 [[attributes]] class=&#x27;next-step...4da4&#x27; [[children]] svg
2. (uid = 815c99ad-b305-4961) [[tag]] span [[xpath]] /html/body.../span[2] [[bbox]] x=715.1 y=280.2 width=153.3 height=44.4 [[attributes]] class=&#x27;input-...4961&#x27; [[children]] span a
3. (uid = 584fe3a7-29a7-4c12) [[tag]] div [[xpath]] /html/body...[1]/div [[bbox]] x=698.1 y=210.8 width=335 height=192.8 [[attributes]] data-v-5...tag-wrapper&#x27; [[children]] span span
4. (uid = 8388af11-580a-4f37) [[tag]] h2 [[xpath]] /html/body...div/h2 [[text]] BRAND KEYWORDS [[bbox]] x=223.7 y=261.6 width=313.6 height=31.0 [[attributes]] data-v-5...f37&#x27;
5. (uid = 542eb267-463d-45ea) [[tag]] a [[xpath]] /html/body/...[2]/a [[bbox]] x=838.3 y=289.6 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27;...45ea&#x27;
6. (uid = d08aa426-c133-4169) [[tag]] input [[xpath]] /html/body...]/div/input [[bbox]] x=715.1 y=333.6 width=301 height=31 [[attributes]] placeholder=&#x27;&#x27; type...4169&#x27;
7. (uid = 1760e1f2-cc86-434d) [[tag]] span [[xpath]] /html/body/...[1]/span [[text]] RunWithFire [[bbox]] x=731.1 y=237.2 width=86.2 height=24 [[attributes]] data-webtasks...34d&#x27;
8. (uid = c147b1a9-054f-4959) [[tag]] span [[xpath]] /html/body...span[1] [[bbox]] x=715.1 y=227.8 width=140.3 height=44.4 [[attributes]] class=&#x27;input-tag...959&#x27; [[children]] span a
9. (uid = 6d5f9829-2d49-4eb1) [[tag]] a [[xpath]] /html/body/...[1]/a [[bbox]] x=825.3 y=237.2 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27;...4eb1&#x27;
10. (uid = f88f49f2-bbd7-41c0) [[tag]] span [[xpath]] /html/body...[2]/span [[text]] Not sure what to use?   [[bbox]] x=223.7 y=358.7 width=184.7 height=24 [[attributes]] data-v-5...1c0&#x27;</pre>

**Gold target (weight)**

<pre>8. (uid = c147b1a9-054f-4959) [[tag]] span [[xpath]] /html/body...span[1] [[bbox]] x=715.1 y=227.8 width=140.3 height=44.4 [[attributes]] class=&#x27;input-tag...959&#x27; [[children]] span a — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · weblinx:train:weblinx:lluitwu:47:weblinx-dom-element-815194ebdc

**State**

<pre>User dialogue:
[00:02] Hello [00:25] Could you please open cheap tickets? [01:08] I am planning a trip to Seychelles for 15th to 22nd July. Could you help me with the things that I can do there? ;
Previous actions:
copy(text=&quot;Rock pool, fantastic &amp; unique experience | Mahé | Private walking tour | NEW&quot;, timestamp=&quot;04:18&quot;) say(speaker=&quot;navigator&quot;, utterance=&quot;Yes, sure. Here are the things that you can do in Seychelles:\n\t1. Cat Cocos: Mahe to Praslin Island Fast Ferry\n\t2.Cat Cocos: Mahe to La Digue Island Fast Ferry\n\t3.Full Day Glass Bottom Boat Excursi...&quot;)&lt;/s&gt;&lt;s&gt;[INST] say(speaker=&quot;instructor&quot;, utterance=&quot;Can you tell me the price for the last option?&quot;) click(uid=&quot;efe80f2b-cc2b-479b&quot;) click(uid=&quot;efe80f2b-cc2b-479b&quot;)&lt;/s&gt;&lt;s&gt;[INST]
DOM:
(html(body class=&quot;&quot;(div(div(div aria-hidden class=&quot;...&quot;(main class(div(div class(div class=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(section class=&quot;...&quot; style=&quot;...&quot;(div aria-label(div class=&quot;...&quot; tabindex data-webtasks-id=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(h4 class=&quot;...&quot; tabindex aria-hidden...)(div class=&quot;...&quot; data-webtasks-id=&quot;...&quot;...)(div class=&quot;...&quot;)(div class=&quot;...&quot;)(section class=&quot;...&quot;(div class=&quot;...&quot;(section data-webtasks-id=&quot;...&quot;(span class=&quot;...&quot;...)(span class=&quot;...&quot; aria-hidden))(div class=&quot;...&quot; aria-hidden(div class=&quot;...&quot; data-webtasks-id=&quot;...&quot;))(div class=&quot;...&quot; aria-hidden))))(div class=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(p class=&quot;...&quot; data-webtasks-id=&quot;...&quot;...) (p class=&quot;...&quot;...) (p class=&quot;...&quot;...))))(div class=&quot;...&quot;(button type aria-label=&quot;...&quot; analytics class=&quot;...&quot; aria-expanded data-webtasks-id=&quot;...&quot;))))(div class=&quot;...&quot;(section class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(section class=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot;(ul class=&quot;...&quot; role(li class=&quot;...&quot; role(ul class=&quot;...&quot; role(li class=&quot;...&quot; role data-webtasks-id=&quot;...&quot;))))))))))))))))))(section class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot; data-webtasks-id=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(h4 class=&quot;...&quot; tabindex)(button type class=&quot;...&quot;))(hr class=&quot;...&quot;(div class=&quot;...&quot;)(div class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(h5 class=&quot;...&quot; data-webtasks-id=&quot;...&quot;...)(div class=&quot;...&quot;)(div class=&quot;...&quot;))(div class=&quot;...&quot; style=&quot;...&quot;(div class=&quot;...&quot;(div class=&quot;...&quot; style=&quot;...&quot;(section class=&quot;...&quot;(div class=&quot;...&quot;(section data-webtasks-id=&quot;...&quot;(span class=&quot;...&quot;...)(span class=&quot;...&quot; aria-hidden...))(div class=&quot;...&quot; aria-hidden)(div class=&quot;...&quot; aria-hidden))))))))(span class=&quot;...&quot; title aria-label aria-hidden)))))))))))))
Operation: click</pre>

**Question**

<pre>Which candidate DOM element should receive the specified operation?</pre>

**Criteria (choice)**

<pre>1. (uid = e8c93b33-f188-455a) [[tag]] p [[xpath]] /html/body/...div/p[1] [[text]] Not everyone who travels...ll take you there.  [[bbox]] x=518.3 y=109.3 width=298.7 height=72 [[attributes]] data-webtasks-...k-paragraph-2&#x27; [[children]] br
2. (uid = 10788327-11f0-4269) [[tag]] li [[xpath]] /html/body/div.../li/ul/li [[text]] Seychelles [[bbox]] x=866.9 y=233.3 width=272.7 height=18 [[attributes]] role=&#x27;listitem&#x27; data...item-size-2&#x27;
3. (uid = b2c34a64-dd01-435e) [[tag]] div [[xpath]] /html/body/div...6]/div/div [[bbox]] x=159.6 y=351.3 width=1016 height=481.3 [[attributes]] id=&#x27;tickets&#x27; data...has-primary-theme&#x27; [[children]] div hr
4. (uid = f816be60-82e9-40a9) [[tag]] button [[xpath]] /html/body/div...div[2]/button [[text]] Show more [[bbox]] x=518.3 y=193.3 width=298.7 height=18 [[attributes]] aria-expanded=&#x27;false...k-link-medium&#x27;
5. (uid = e22ab097-37db-42bb) [[tag]] section [[xpath]] /html/body/div...div[2]/section [[bbox]] x=195.6 y=121.3 width=298.7 height=26 [[attributes]] data-webtasks-...db-42bb&#x27; [[children]] span span
6. (uid = fb22e5f9-29da-44b9) [[tag]] div [[xpath]] /html/body/div...div[1]/div [[text]] per adult* [[bbox]] x=195.6 y=147.3 width=298.7 height=12.9 [[attributes]] class=&#x27;uitk-price...44b9&#x27;
7. (uid = a3ea9114-2e62-45a3) [[tag]] div [[xpath]] /html/body/...1]/div[1] [[text]] By Island Tour Seychelles [[bbox]] x=195.6 y=-12.7 width=298.7 height=18 [[attributes]] data-webtasks-id...-blockstart-two&#x27;
8. (uid = 01539f28-81b7-4fe3) [[tag]] h5 [[xpath]] /html/body/...div[1]/h5 [[text]] Rock pool, fantastic... | Private walking tour | NEW [[bbox]] x=195.6 y=526.1 width=298.7 height=66 [[attributes]] data-webtasks-id...heading-5&#x27;
9. (uid = c1947059-f645-4734) [[tag]] div [[xpath]] /html/body/div...4]/div/div [[bbox]] x=159.6 y=-134.7 width=1016 height=462 [[attributes]] tabindex=&#x27;-1&#x27;...-theme-primary&#x27; [[children]] div
10. (uid = c92bdda7-c84f-4fe1) [[tag]] section [[xpath]] /html/body/div...section/div/section [[bbox]] x=883.7 y=610.9 width=255.9 height=26 [[attributes]] data-webtasks-id...-4fe1&#x27; [[children]] span span</pre>

**Gold target (weight)**

<pre>10. (uid = c92bdda7-c84f-4fe1) [[tag]] section [[xpath]] /html/body/div...section/div/section [[bbox]] x=883.7 y=610.9 width=255.9 height=26 [[attributes]] data-webtasks-id...-4fe1&#x27; [[children]] span span — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · weblinx:train:weblinx:hoxokpu:23:weblinx-dom-element-815194ebdc

**State**

<pre>User dialogue:
[-00:06] Hi [00:13] Could you please create a logo for my online women’s designer brand on brandmark.io? [01:13] Enchanté Boutique [01:37] Yes. Please add &quot;Beauty at its best.&quot; ;
Previous actions:
click(uid=&quot;505a6898-e589-4649&quot;) paste(text=&quot;Beauty at its best.&quot;, timestamp=&quot;01:43&quot;) text_input(text=&quot;Beauty at its best&quot;, uid=&quot;505a6898-e589-4649&quot;) click(uid=&quot;907efcdb-5ac0-44db&quot;) click(uid=&quot;1a01252c-d493-4256&quot;)&lt;/s&gt;&lt;s&gt;[INST]
DOM:
(html(div(div class=&quot;&quot;(div class=&quot;wizard-panel keywords-wizard-panel active&quot;(div class=&quot;wizard-card&quot;(div class=&quot;wizard-card-content&quot;(div(div class=&quot;vue-input-tag-wrapper&quot; data-webtasks-id=&quot;fed386c2-1a18-4338&quot;(span class=&quot;input-tag&quot; data-webtasks-id=&quot;4183ab2f-9c08-4ee7&quot;(span data-webtasks-id=&quot;089d038e-8746-4e15&quot;Paper)(a class=&quot;remove&quot; data-webtasks-id=&quot;03061b56-f513-4f4c&quot;))(span class=&quot;input-tag&quot; data-webtasks-id=&quot;44f47efa-b8a6-4cd1&quot;(a class=&quot;remove&quot; data-webtasks-id=&quot;1a01252c-d493-4256&quot;))(span class=&quot;input-tag&quot;(a class=&quot;remove&quot; data-webtasks-id=&quot;e9ccf138-48b8-476b&quot;))(input placeholder=&quot;&quot; type=&quot;text&quot; class=&quot;new-tag&quot; spellcheck=&quot;false&quot; data-webtasks-id=&quot;afb6bfba-a41d-4aea&quot;)))))(div class=&quot;wizard-panel next&quot;)(div class=&quot;wizard-panel next&quot;)(button class=&quot;next-step-btn&quot; data-webtasks-id=&quot;42cea898-b9e3-4216&quot;(svg width=&quot;25.805&quot; height=&quot;44.891&quot; viewbox=&quot;0 0 25.805 44.891&quot;(path d=&quot;M8806.227,558.43l8.188-8.189c10.092-10.091,10.926-11.013,10.926-11.013s-.834-.788-10.976-10.93l-8.23...65-10.831,10.832c-5.109,5.109-9.016,9-10.912,10.875C8808.4,560.6,8806.227,558.43,8806.227,558.43Z&quot; transform=&quot;translate(-8805.419 -516.783)&quot; fill=&quot;#fff&quot; stroke=&quot;#fff&quot; stroke-miterlimit=&quot;10&quot; stroke-width=&quot;1&quot; data-webtasks-id=&quot;907efcdb-5ac0-44db&quot;))))))
Operation: click</pre>

**Question**

<pre>Which candidate DOM element should receive the specified operation?</pre>

**Criteria (choice)**

<pre>1. (uid = 907efcdb-5ac0-44db) [[tag]] path [[xpath]] /html/body/...button/svg/path [[bbox]] x=1249.0 y=351.1 width=24.4 height=43.5 [[attributes]] stroke-width=&#x27;1...8.43Z&#x27;
2. (uid = 03061b56-f513-4f4c) [[tag]] a [[xpath]] /html/body/...span[1]/a [[bbox]] x=863.8 y=323.7 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27; data...4f4c&#x27;
3. (uid = 44f47efa-b8a6-4cd1) [[tag]] span [[xpath]] /html/body/.../span[2] [[bbox]] x=901.9 y=314.9 width=96.0 height=44 [[attributes]] class=&#x27;input-tag&#x27;...-4cd1&#x27; [[children]] span a
4. (uid = 42cea898-b9e3-4216) [[tag]] button [[xpath]] /html/body/div/div/button [[bbox]] x=1186 y=85.3 width=132 height=575 [[attributes]] class=&#x27;next-step...-4216&#x27; [[children]] svg
5. (uid = 089d038e-8746-4e15) [[tag]] span [[xpath]] /html/body/div...span[1]/span [[text]] Paper [[bbox]] x=815.9 y=323.7 width=39.9 height=24 [[attributes]] data-webtasks-...4e15&#x27;
6. (uid = fed386c2-1a18-4338) [[tag]] div [[xpath]] /html/body/...div[1]/div [[bbox]] x=783.1 y=298.1 width=335 height=153.6 [[attributes]] data-v-50...-tag-wrapper&#x27; [[children]] span span
7. (uid = 1a01252c-d493-4256) [[tag]] a [[xpath]] /html/body/...span[2]/a [[bbox]] x=967.8 y=323.7 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27; data...4256&#x27;
8. (uid = 4183ab2f-9c08-4ee7) [[tag]] span [[xpath]] /html/body/.../span[1] [[bbox]] x=799.9 y=314.9 width=94.0 height=44 [[attributes]] class=&#x27;input-tag...-4ee7&#x27; [[children]] span a
9. (uid = afb6bfba-a41d-4aea) [[tag]] input [[xpath]] /html/body/...1]/div/input [[bbox]] x=904.3 y=367.9 width=197.0 height=45 [[attributes]] placeholder=&#x27;&#x27; type=&#x27;text...-4aea&#x27;
10. (uid = e9ccf138-48b8-476b) [[tag]] a [[xpath]] /html/body/...span[3]/a [[bbox]] x=866.2 y=375.7 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27; data...476b&#x27;</pre>

**Gold target (weight)**

<pre>7. (uid = 1a01252c-d493-4256) [[tag]] a [[xpath]] /html/body/...span[2]/a [[bbox]] x=967.8 y=323.7 width=14.1 height=24 [[attributes]] class=&#x27;remove&#x27; data...4256&#x27; — 1</pre>

</details>
