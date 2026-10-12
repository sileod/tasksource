# stackoverflow-questions

Classification · `pacovaldez/stackoverflow-questions`

<details><summary>Example 1</summary>

published Jev · train · stackoverflow-questions-2f8d07e288:train:16751

**State**

<pre>text_A: Is there a way to discover the original generic argument type of a instance of an Action&lt;in T&gt; delegate?
text_B: &lt;p&gt;Is there a way to find the declared generic argument type of an instance of an Action delegate?&lt;/p&gt;

&lt;p&gt;In the following code the &lt;code&gt;WriteGenericArgumentType&lt;/code&gt; function expects and instance of &lt;code&gt;Action&lt;String&gt;&lt;/code&gt; but because &lt;code&gt;Action&lt;in T&gt;&lt;/code&gt; allows for contravariance callers of the function are able to call it with &lt;code&gt;Action&lt;Object&gt;&lt;/code&gt;.  &lt;/p&gt;

&lt;pre&gt;&lt;code&gt;static void Main( )
{
    WriteGenericArgumentType( new Action&lt;string&gt;( s =&gt; { } ) );
    WriteGenericArgumentType( new Action&lt;object&gt;( o =&gt; { } ) );
}

static void WriteGenericArgumentType( Action&lt;string&gt; action )
{
    Console.WriteLine( DiscoverGenericArgumentType( action ).Name );
}

static Type DiscoverGenericArgumentType( Delegate action )
{
    return action.GetType().GetGenericArguments()[0];
}
&lt;/code&gt;&lt;/pre&gt;

&lt;p&gt;However, I need to be able to discover the generic argument type that was defined in the signature of &lt;code&gt;WriteGenericArgumentType&lt;/code&gt; from the instance of the delegate that is passed to &lt;code&gt;DiscoverGenericArgumentType&lt;/code&gt;.&lt;/p&gt;

&lt;p&gt;When the code runs I get the following output:&lt;/p&gt;

&lt;pre&gt;&lt;code&gt;String
Object
&lt;/code&gt;&lt;/pre&gt;

&lt;p&gt;But I need it to be:&lt;/p&gt;

&lt;pre&gt;&lt;code&gt;String
String
&lt;/code&gt;&lt;/pre&gt;

&lt;p&gt;For my purposes I cannot change the signature of &lt;code&gt;DiscoverGenericArgumentType&lt;/code&gt; and I need that function to return &lt;code&gt;string&lt;/code&gt; as the type.&lt;/p&gt;

&lt;p&gt;I know I can create my own generic delegate type that doesn&#x27;t allow contravariance but I&#x27;m looking for a way to not have to change the signature of the API if possible.&lt;/p&gt;

&lt;p&gt;Any ideas?&lt;/p&gt;</pre>

**Question**

<pre>How popular is this Stack Overflow question?</pre>

**Criteria (choice)**

<pre>1. very popular question
2. popular question
3. somewhat popular question
4. unpopular question</pre>

**Gold target (weight)**

<pre>4. unpopular question — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · stackoverflow-questions-2f8d07e288:train:2486

**State**

<pre>text_A: Non-const implicit &quot;converting&quot; constructor
text_B: &lt;p&gt;Hi guys I&#x27;m working on a messaging system where there are 3 types: Streams (memory), Taps (const view type), Sink(mutable view type). Now constructing a tap from a stream is very easy because it can implicitly convert from a stream to a tap by using the standard const ref conversion constructor. But for sinks I can&#x27;t seem to figure out how to make it implicitly convert. Implicit conversion is something that saves me a lot of headaches (I&#x27;m not trying to get into the whole implicit conversion is evil discussion here) working with tuple and constexpr code here, would it be possible to implicitly construct a sink from a stream, maintaining const correctness?    &lt;/p&gt;

&lt;pre&gt;&lt;code&gt;#include &lt;tuple&gt;

template&lt;typename T&gt;
struct Stream
{
};

template&lt;typename T&gt;
struct Tap
{
    Tap(const Stream&lt;T&gt;&amp; stream)
    {}
};

template&lt;typename T&gt;
struct Sink
{
    Sink(Stream&lt;T&gt;&amp;)
    {}
};



int main()
{
    Stream&lt;float&gt; stream;
    Tap&lt;float&gt; tap = stream;
    //Sink&lt;float&gt; sink = stream;

    std::tuple&lt;Stream&lt;float&gt;, Stream&lt;int&gt;&gt; streams;
    std::tuple&lt;Tap&lt;float&gt;, Tap&lt;int&gt;&gt; taps = streams;
    //std::tuple&lt;Sink&lt;float&gt;, Sink&lt;int&gt;&gt; sinks = streams;
}
&lt;/code&gt;&lt;/pre&gt;</pre>

**Question**

<pre>How popular is this Stack Overflow question?</pre>

**Criteria (choice)**

<pre>1. very popular question
2. popular question
3. somewhat popular question
4. unpopular question</pre>

**Gold target (weight)**

<pre>4. unpopular question — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · stackoverflow-questions-2f8d07e288:train:27470

**State**

<pre>A: How to fix &#x27;map center &amp; render all tiles&#x27; with vue-leaflet
B: &lt;p&gt;&lt;strong&gt;Error loading tiles maps&lt;/strong&gt;&lt;/p&gt;

&lt;p&gt;the page loads but the map is not centered completely and tiles are missing even if I already put the props :center &amp; :zoom&lt;/p&gt;

&lt;p&gt;can I initialize the map when loading the page?&lt;/p&gt;

&lt;p&gt;&lt;a href=&quot;https://i.stack.imgur.com/emV5U.png&quot; rel=&quot;nofollow noreferrer&quot;&gt;Vue-Leaflet&lt;/a&gt;&lt;/p&gt;

&lt;p&gt;this is my code
&lt;div class=&quot;snippet&quot; data-lang=&quot;js&quot; data-hide=&quot;false&quot; data-console=&quot;true&quot; data-babel=&quot;false&quot;&gt;
&lt;div class=&quot;snippet-code&quot;&gt;
&lt;pre class=&quot;snippet-code-js lang-js prettyprint-override&quot;&gt;&lt;code&gt;&lt;template&gt;
    &lt;div class=&quot;map&quot;&gt;
        &lt;l-map class=&quot;maps&quot; :zoom=&quot;zoom&quot; :center=&quot;center&quot;&gt;
            &lt;l-marker :lat-lng=&quot;marker&quot;&gt;&lt;/l-marker&gt;
            &lt;l-tile-layer :url=&quot;url&quot; :attribution=&quot;attribution&quot;&gt;&lt;/l-tile-layer&gt;
        &lt;/l-map&gt;
    &lt;/div&gt;
&lt;/template&gt;

&lt;script&gt;
import { L, LMap, LTileLayer, LMarker } from &#x27;vue2-leaflet&#x27;;
import { Icon } from &#x27;leaflet&#x27;;
import &#x27;leaflet/dist/leaflet.css&#x27;

delete Icon.Default.prototype._getIconUrl;
Icon.Default.mergeOptions({
    iconRetinaUrl: require(&#x27;leaflet/dist/images/marker-icon-2x.png&#x27;),
    iconUrl: require(&#x27;leaflet/dist/images/marker-icon.png&#x27;),
    shadowUrl: require(&#x27;leaflet/dist/images/marker-shadow.png&#x27;)
});
export default {
    props:[&#x27;lat&#x27;,&#x27;lng&#x27;],
    mounted(){
        // this.center = L.latLng(this.lat,this.lng)
        // this.marker = L.latLng(this.lat,this.lng)
    },
    data(){
        return{
            zoom:13,
            center: L.latLng(this.lat,this.lng),
            url:&#x27;http://{s}.tile.osm.org/{z}/{x}/{y}.png&#x27;,
            attribution:&#x27;&amp;copy; &lt;a href=&quot;http://osm.org/copyright&quot;&gt;OpenStreetMap&lt;/a&gt; contributors&#x27;,
            marker: L.latLng(this.lat,this.lng),
        }
    },
    components:{
        LMap, LTileLayer, LMarker
    }
}
&lt;/script&gt;

&lt;style&gt;
.map{
    width: 100%;
    height: 500px;
}
&lt;/style&gt;&lt;/code&gt;&lt;/pre&gt;
&lt;/div&gt;
&lt;/div&gt;
&lt;/p&gt;</pre>

**Question**

<pre>How popular is this Stack Overflow question?</pre>

**Criteria (score)**

<pre>1. very popular question
2. popular question
3. somewhat popular question
4. unpopular question</pre>

**Gold target (weight)**

<pre>3. somewhat popular question — 1</pre>

</details>
