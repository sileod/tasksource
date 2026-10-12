# code_x_glue_cc_defect_detection

Classification · `google/code_x_glue_cc_defect_detection`

<details><summary>Example 1</summary>

published Jev · train · code-x-glue-cc-defect-detection-1c76c57e72:train:1

**State**

<pre>static int transcode(AVFormatContext **output_files,

                     int nb_output_files,

                     InputFile *input_files,

                     int nb_input_files,

                     StreamMap *stream_maps, int nb_stream_maps)

{

    int ret = 0, i, j, k, n, nb_ostreams = 0, step;



    AVFormatContext *is, *os;

    AVCodecContext *codec, *icodec;

    OutputStream *ost, **ost_table = NULL;

    InputStream *ist;

    char error[1024];

    int key;

    int want_sdp = 1;

    uint8_t no_packet[MAX_FILES]={0};

    int no_packet_count=0;

    int nb_frame_threshold[AVMEDIA_TYPE_NB]={0};

    int nb_streams[AVMEDIA_TYPE_NB]={0};



    if (rate_emu)

        for (i = 0; i &lt; nb_input_streams; i++)

            input_streams[i].start = av_gettime();



    /* output stream init */

    nb_ostreams = 0;

    for(i=0;i&lt;nb_output_files;i++) {

        os = output_files[i];

        if (!os-&gt;nb_streams &amp;&amp; !(os-&gt;oformat-&gt;flags &amp; AVFMT_NOSTREAMS)) {

            av_dump_format(output_files[i], i, output_files[i]-&gt;filename, 1);

            fprintf(stderr, &quot;Output file #%d does not contain any stream\n&quot;, i);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        nb_ostreams += os-&gt;nb_streams;

    }

    if (nb_stream_maps &gt; 0 &amp;&amp; nb_stream_maps != nb_ostreams) {

        fprintf(stderr, &quot;Number of stream maps must match number of output streams\n&quot;);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    /* Sanity check the mapping args -- do the input files &amp; streams exist? */

    for(i=0;i&lt;nb_stream_maps;i++) {

        int fi = stream_maps[i].file_index;

        int si = stream_maps[i].stream_index;



        if (fi &lt; 0 || fi &gt; nb_input_files - 1 ||

            si &lt; 0 || si &gt; input_files[fi].ctx-&gt;nb_streams - 1) {

            fprintf(stderr,&quot;Could not find input stream #%d.%d\n&quot;, fi, si);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        fi = stream_maps[i].sync_file_index;

        si = stream_maps[i].sync_stream_index;

        if (fi &lt; 0 || fi &gt; nb_input_files - 1 ||

            si &lt; 0 || si &gt; input_files[fi].ctx-&gt;nb_streams - 1) {

            fprintf(stderr,&quot;Could not find sync stream #%d.%d\n&quot;, fi, si);

            ret = AVERROR(EINVAL);

            goto fail;

        }

    }



    ost_table = av_mallocz(sizeof(OutputStream *) * nb_ostreams);

    if (!ost_table)

        goto fail;



    for(k=0;k&lt;nb_output_files;k++) {

        os = output_files[k];

        for(i=0;i&lt;os-&gt;nb_streams;i++,n++) {

            nb_streams[os-&gt;streams[i]-&gt;codec-&gt;codec_type]++;

        }

    }

    for(step=1&lt;&lt;30; step; step&gt;&gt;=1){

        int found_streams[AVMEDIA_TYPE_NB]={0};

        for(j=0; j&lt;AVMEDIA_TYPE_NB; j++)

            nb_frame_threshold[j] += step;



        for(j=0; j&lt;nb_input_streams; j++) {

            int skip=0;

            ist = &amp;input_streams[j];

            if(opt_programid){

                int pi,si;

                AVFormatContext *f= input_files[ ist-&gt;file_index ].ctx;

                skip=1;

                for(pi=0; pi&lt;f-&gt;nb_programs; pi++){

                    AVProgram *p= f-&gt;programs[pi];

                    if(p-&gt;id == opt_programid)

                        for(si=0; si&lt;p-&gt;nb_stream_indexes; si++){

                            if(f-&gt;streams[ p-&gt;stream_index[si] ] == ist-&gt;st)

                                skip=0;

                        }

                }

            }

            if (ist-&gt;discard &amp;&amp; ist-&gt;st-&gt;discard != AVDISCARD_ALL &amp;&amp; !skip

                &amp;&amp; nb_frame_threshold[ist-&gt;st-&gt;codec-&gt;codec_type] &lt;= ist-&gt;st-&gt;codec_info_nb_frames){

                found_streams[ist-&gt;st-&gt;codec-&gt;codec_type]++;

            }

        }

        for(j=0; j&lt;AVMEDIA_TYPE_NB; j++)

            if(found_streams[j] &lt; nb_streams[j])

                nb_frame_threshold[j] -= step;

    }

    n = 0;

    for(k=0;k&lt;nb_output_files;k++) {

        os = output_files[k];

        for(i=0;i&lt;os-&gt;nb_streams;i++,n++) {

            int found;

            ost = ost_table[n] = output_streams_for_file[k][i];

            if (nb_stream_maps &gt; 0) {

                ost-&gt;source_index = input_files[stream_maps[n].file_index].ist_index +

                    stream_maps[n].stream_index;



                /* Sanity check that the stream types match */

                if (input_streams[ost-&gt;source_index].st-&gt;codec-&gt;codec_type != ost-&gt;st-&gt;codec-&gt;codec_type) {

                    int i= ost-&gt;file_index;

                    av_dump_format(output_files[i], i, output_files[i]-&gt;filename, 1);

                    fprintf(stderr, &quot;Codec type mismatch for mapping #%d.%d -&gt; #%d.%d\n&quot;,

                        stream_maps[n].file_index, stream_maps[n].stream_index,

                        ost-&gt;file_index, ost-&gt;index);

                    ffmpeg_exit(1);

                }



            } else {

                /* get corresponding input stream index : we select the first one with the right type */

                found = 0;

                for (j = 0; j &lt; nb_input_streams; j++) {

                    int skip=0;

                    ist = &amp;input_streams[j];

                    if(opt_programid){

                        int pi,si;

                        AVFormatContext *f = input_files[ist-&gt;file_index].ctx;

                        skip=1;

                        for(pi=0; pi&lt;f-&gt;nb_programs; pi++){

                            AVProgram *p= f-&gt;programs[pi];

                            if(p-&gt;id == opt_programid)

                                for(si=0; si&lt;p-&gt;nb_stream_indexes; si++){

                                    if(f-&gt;streams[ p-&gt;stream_index[si] ] == ist-&gt;st)

                                        skip=0;

                                }

                        }

                    }

                    if (ist-&gt;discard &amp;&amp; ist-&gt;st-&gt;discard != AVDISCARD_ALL &amp;&amp; !skip &amp;&amp;

                        ist-&gt;st-&gt;codec-&gt;codec_type == ost-&gt;st-&gt;codec-&gt;codec_type &amp;&amp;

                        nb_frame_threshold[ist-&gt;st-&gt;codec-&gt;codec_type] &lt;= ist-&gt;st-&gt;codec_info_nb_frames) {

                            ost-&gt;source_index = j;

                            found = 1;

                            break;

                    }

                }



                if (!found) {

                    if(! opt_programid) {

                        /* try again and reuse existing stream */

                        for (j = 0; j &lt; nb_input_streams; j++) {

                            ist = &amp;input_streams[j];

                            if (   ist-&gt;st-&gt;codec-&gt;codec_type == ost-&gt;st-&gt;codec-&gt;codec_type

                                &amp;&amp; ist-&gt;st-&gt;discard != AVDISCARD_ALL) {

                                ost-&gt;source_index = j;

                                found = 1;

                            }

                        }

                    }

                    if (!found) {

                        int i= ost-&gt;file_index;

                        av_dump_format(output_files[i], i, output_files[i]-&gt;filename, 1);

                        fprintf(stderr, &quot;Could not find input stream matching output stream #%d.%d\n&quot;,

                                ost-&gt;file_index, ost-&gt;index);

                        ffmpeg_exit(1);

                    }

                }

            }

            ist = &amp;input_streams[ost-&gt;source_index];

            ist-&gt;discard = 0;

            ost-&gt;sync_ist = (nb_stream_maps &gt; 0) ?

                &amp;input_streams[input_files[stream_maps[n].sync_file_index].ist_index +

                         stream_maps[n].sync_stream_index] : ist;

        }

    }



    /* for each output stream, we compute the right encoding parameters */

    for(i=0;i&lt;nb_ostreams;i++) {

        ost = ost_table[i];

        os = output_files[ost-&gt;file_index];

        ist = &amp;input_streams[ost-&gt;source_index];



        codec = ost-&gt;st-&gt;codec;

        icodec = ist-&gt;st-&gt;codec;



        if (metadata_streams_autocopy)

            av_dict_copy(&amp;ost-&gt;st-&gt;metadata, ist-&gt;st-&gt;metadata,

                         AV_DICT_DONT_OVERWRITE);



        ost-&gt;st-&gt;disposition = ist-&gt;st-&gt;disposition;

        codec-&gt;bits_per_raw_sample= icodec-&gt;bits_per_raw_sample;

        codec-&gt;chroma_sample_location = icodec-&gt;chroma_sample_location;



        if (ost-&gt;st-&gt;stream_copy) {

            uint64_t extra_size = (uint64_t)icodec-&gt;extradata_size + FF_INPUT_BUFFER_PADDING_SIZE;



            if (extra_size &gt; INT_MAX)

                goto fail;



            /* if stream_copy is selected, no need to decode or encode */

            codec-&gt;codec_id = icodec-&gt;codec_id;

            codec-&gt;codec_type = icodec-&gt;codec_type;



            if(!codec-&gt;codec_tag){

                if(   !os-&gt;oformat-&gt;codec_tag

                   || av_codec_get_id (os-&gt;oformat-&gt;codec_tag, icodec-&gt;codec_tag) == codec-&gt;codec_id

                   || av_codec_get_tag(os-&gt;oformat-&gt;codec_tag, icodec-&gt;codec_id) &lt;= 0)

                    codec-&gt;codec_tag = icodec-&gt;codec_tag;

            }



            codec-&gt;bit_rate = icodec-&gt;bit_rate;

            codec-&gt;rc_max_rate    = icodec-&gt;rc_max_rate;

            codec-&gt;rc_buffer_size = icodec-&gt;rc_buffer_size;

            codec-&gt;extradata= av_mallocz(extra_size);

            if (!codec-&gt;extradata)

                goto fail;

            memcpy(codec-&gt;extradata, icodec-&gt;extradata, icodec-&gt;extradata_size);

            codec-&gt;extradata_size= icodec-&gt;extradata_size;

            if(!copy_tb &amp;&amp; av_q2d(icodec-&gt;time_base)*icodec-&gt;ticks_per_frame &gt; av_q2d(ist-&gt;st-&gt;time_base) &amp;&amp; av_q2d(ist-&gt;st-&gt;time_base) &lt; 1.0/500){

                codec-&gt;time_base = icodec-&gt;time_base;

                codec-&gt;time_base.num *= icodec-&gt;ticks_per_frame;

                av_reduce(&amp;codec-&gt;time_base.num, &amp;codec-&gt;time_base.den,

                          codec-&gt;time_base.num, codec-&gt;time_base.den, INT_MAX);

            }else

                codec-&gt;time_base = ist-&gt;st-&gt;time_base;

            switch(codec-&gt;codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                if(audio_volume != 256) {

                    fprintf(stderr,&quot;-acodec copy and -vol are incompatible (frames are not decoded)\n&quot;);

                    ffmpeg_exit(1);

                }

                codec-&gt;channel_layout = icodec-&gt;channel_layout;

                codec-&gt;sample_rate = icodec-&gt;sample_rate;

                codec-&gt;channels = icodec-&gt;channels;

                codec-&gt;frame_size = icodec-&gt;frame_size;

                codec-&gt;audio_service_type = icodec-&gt;audio_service_type;

                codec-&gt;block_align= icodec-&gt;block_align;

                if(codec-&gt;block_align == 1 &amp;&amp; codec-&gt;codec_id == CODEC_ID_MP3)

                    codec-&gt;block_align= 0;

                if(codec-&gt;codec_id == CODEC_ID_AC3)

                    codec-&gt;block_align= 0;

                break;

            case AVMEDIA_TYPE_VIDEO:

                codec-&gt;pix_fmt = icodec-&gt;pix_fmt;

                codec-&gt;width = icodec-&gt;width;

                codec-&gt;height = icodec-&gt;height;

                codec-&gt;has_b_frames = icodec-&gt;has_b_frames;

                if (!codec-&gt;sample_aspect_ratio.num) {

                    codec-&gt;sample_aspect_ratio =

                    ost-&gt;st-&gt;sample_aspect_ratio =

                        ist-&gt;st-&gt;sample_aspect_ratio.num ? ist-&gt;st-&gt;sample_aspect_ratio :

                        ist-&gt;st-&gt;codec-&gt;sample_aspect_ratio.num ?

                        ist-&gt;st-&gt;codec-&gt;sample_aspect_ratio : (AVRational){0, 1};

                }

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                codec-&gt;width = icodec-&gt;width;

                codec-&gt;height = icodec-&gt;height;

                break;

            case AVMEDIA_TYPE_DATA:

                break;

            default:

                abort();

            }

        } else {

            if (!ost-&gt;enc)

                ost-&gt;enc = avcodec_find_encoder(ost-&gt;st-&gt;codec-&gt;codec_id);

            switch(codec-&gt;codec_type) {

            case AVMEDIA_TYPE_AUDIO:

                ost-&gt;fifo= av_fifo_alloc(1024);

                if(!ost-&gt;fifo)

                    goto fail;

                ost-&gt;reformat_pair = MAKE_SFMT_PAIR(AV_SAMPLE_FMT_NONE,AV_SAMPLE_FMT_NONE);

                if (!codec-&gt;sample_rate) {

                    codec-&gt;sample_rate = icodec-&gt;sample_rate;

                    if (icodec-&gt;lowres)

                        codec-&gt;sample_rate &gt;&gt;= icodec-&gt;lowres;

                }

                choose_sample_rate(ost-&gt;st, ost-&gt;enc);

                codec-&gt;time_base = (AVRational){1, codec-&gt;sample_rate};

                if (codec-&gt;sample_fmt == AV_SAMPLE_FMT_NONE)

                    codec-&gt;sample_fmt = icodec-&gt;sample_fmt;

                choose_sample_fmt(ost-&gt;st, ost-&gt;enc);

                if (!codec-&gt;channels) {

                    codec-&gt;channels = icodec-&gt;channels;

                    codec-&gt;channel_layout = icodec-&gt;channel_layout;

                }

                if (av_get_channel_layout_nb_channels(codec-&gt;channel_layout) != codec-&gt;channels)

                    codec-&gt;channel_layout = 0;

                ost-&gt;audio_resample = codec-&gt;sample_rate != icodec-&gt;sample_rate || audio_sync_method &gt; 1;

                icodec-&gt;request_channels = codec-&gt;channels;

                ist-&gt;decoding_needed = 1;

                ost-&gt;encoding_needed = 1;

                ost-&gt;resample_sample_fmt  = icodec-&gt;sample_fmt;

                ost-&gt;resample_sample_rate = icodec-&gt;sample_rate;

                ost-&gt;resample_channels    = icodec-&gt;channels;

                break;

            case AVMEDIA_TYPE_VIDEO:

                if (codec-&gt;pix_fmt == PIX_FMT_NONE)

                    codec-&gt;pix_fmt = icodec-&gt;pix_fmt;

                choose_pixel_fmt(ost-&gt;st, ost-&gt;enc);



                if (ost-&gt;st-&gt;codec-&gt;pix_fmt == PIX_FMT_NONE) {

                    fprintf(stderr, &quot;Video pixel format is unknown, stream cannot be encoded\n&quot;);

                    ffmpeg_exit(1);

                }

                ost-&gt;video_resample = codec-&gt;width   != icodec-&gt;width  ||

                                      codec-&gt;height  != icodec-&gt;height ||

                                      codec-&gt;pix_fmt != icodec-&gt;pix_fmt;

                if (ost-&gt;video_resample) {

                    codec-&gt;bits_per_raw_sample= frame_bits_per_raw_sample;

                }

                if (!codec-&gt;width || !codec-&gt;height) {

                    codec-&gt;width  = icodec-&gt;width;

                    codec-&gt;height = icodec-&gt;height;

                }

                ost-&gt;resample_height = icodec-&gt;height;

                ost-&gt;resample_width  = icodec-&gt;width;

                ost-&gt;resample_pix_fmt= icodec-&gt;pix_fmt;

                ost-&gt;encoding_needed = 1;

                ist-&gt;decoding_needed = 1;



                if (!ost-&gt;frame_rate.num)

                    ost-&gt;frame_rate = ist-&gt;st-&gt;r_frame_rate.num ? ist-&gt;st-&gt;r_frame_rate : (AVRational){25,1};

                if (ost-&gt;enc &amp;&amp; ost-&gt;enc-&gt;supported_framerates &amp;&amp; !force_fps) {

                    int idx = av_find_nearest_q_idx(ost-&gt;frame_rate, ost-&gt;enc-&gt;supported_framerates);

                    ost-&gt;frame_rate = ost-&gt;enc-&gt;supported_framerates[idx];

                }

                codec-&gt;time_base = (AVRational){ost-&gt;frame_rate.den, ost-&gt;frame_rate.num};

                if(   av_q2d(codec-&gt;time_base) &lt; 0.001 &amp;&amp; video_sync_method

                   &amp;&amp; (video_sync_method==1 || (video_sync_method&lt;0 &amp;&amp; !(os-&gt;oformat-&gt;flags &amp; AVFMT_VARIABLE_FPS)))){

                    av_log(os, AV_LOG_WARNING, &quot;Frame rate very high for a muxer not effciciently supporting it.\n&quot;

                                               &quot;Please consider specifiying a lower framerate, a different muxer or -vsync 2\n&quot;);

                }



#if CONFIG_AVFILTER

                if (configure_video_filters(ist, ost)) {

                    fprintf(stderr, &quot;Error opening filters!\n&quot;);

                    exit(1);

                }

#endif

                break;

            case AVMEDIA_TYPE_SUBTITLE:

                ost-&gt;encoding_needed = 1;

                ist-&gt;decoding_needed = 1;

                break;

            default:

                abort();

                break;

            }

            /* two pass mode */

            if (ost-&gt;encoding_needed &amp;&amp; codec-&gt;codec_id != CODEC_ID_H264 &amp;&amp;

                (codec-&gt;flags &amp; (CODEC_FLAG_PASS1 | CODEC_FLAG_PASS2))) {

                char logfilename[1024];

                FILE *f;



                snprintf(logfilename, sizeof(logfilename), &quot;%s-%d.log&quot;,

                         pass_logfilename_prefix ? pass_logfilename_prefix : DEFAULT_PASS_LOGFILENAME_PREFIX,

                         i);

                if (codec-&gt;flags &amp; CODEC_FLAG_PASS1) {

                    f = fopen(logfilename, &quot;wb&quot;);

                    if (!f) {

                        fprintf(stderr, &quot;Cannot write log file &#x27;%s&#x27; for pass-1 encoding: %s\n&quot;, logfilename, strerror(errno));

                        ffmpeg_exit(1);

                    }

                    ost-&gt;logfile = f;

                } else {

                    char  *logbuffer;

                    size_t logbuffer_size;

                    if (read_file(logfilename, &amp;logbuffer, &amp;logbuffer_size) &lt; 0) {

                        fprintf(stderr, &quot;Error reading log file &#x27;%s&#x27; for pass-2 encoding\n&quot;, logfilename);

                        ffmpeg_exit(1);

                    }

                    codec-&gt;stats_in = logbuffer;

                }

            }

        }

        if(codec-&gt;codec_type == AVMEDIA_TYPE_VIDEO){

            /* maximum video buffer size is 6-bytes per pixel, plus DPX header size */

            int size= codec-&gt;width * codec-&gt;height;

            bit_buffer_size= FFMAX(bit_buffer_size, 6*size + 1664);

        }

    }



    if (!bit_buffer)

        bit_buffer = av_malloc(bit_buffer_size);

    if (!bit_buffer) {

        fprintf(stderr, &quot;Cannot allocate %d bytes output buffer\n&quot;,

                bit_buffer_size);

        ret = AVERROR(ENOMEM);

        goto fail;

    }



    /* open each encoder */

    for(i=0;i&lt;nb_ostreams;i++) {

        ost = ost_table[i];

        if (ost-&gt;encoding_needed) {

            AVCodec *codec = ost-&gt;enc;

            AVCodecContext *dec = input_streams[ost-&gt;source_index].st-&gt;codec;

            if (!codec) {

                snprintf(error, sizeof(error), &quot;Encoder (codec id %d) not found for output stream #%d.%d&quot;,

                         ost-&gt;st-&gt;codec-&gt;codec_id, ost-&gt;file_index, ost-&gt;index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            if (dec-&gt;subtitle_header) {

                ost-&gt;st-&gt;codec-&gt;subtitle_header = av_malloc(dec-&gt;subtitle_header_size);

                if (!ost-&gt;st-&gt;codec-&gt;subtitle_header) {

                    ret = AVERROR(ENOMEM);

                    goto dump_format;

                }

                memcpy(ost-&gt;st-&gt;codec-&gt;subtitle_header, dec-&gt;subtitle_header, dec-&gt;subtitle_header_size);

                ost-&gt;st-&gt;codec-&gt;subtitle_header_size = dec-&gt;subtitle_header_size;

            }

            if (avcodec_open2(ost-&gt;st-&gt;codec, codec, &amp;ost-&gt;opts) &lt; 0) {

                snprintf(error, sizeof(error), &quot;Error while opening encoder for output stream #%d.%d - maybe incorrect parameters such as bit_rate, rate, width or height&quot;,

                        ost-&gt;file_index, ost-&gt;index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            assert_codec_experimental(ost-&gt;st-&gt;codec, 1);

            assert_avoptions(ost-&gt;opts);

            if (ost-&gt;st-&gt;codec-&gt;bit_rate &amp;&amp; ost-&gt;st-&gt;codec-&gt;bit_rate &lt; 1000)

                av_log(NULL, AV_LOG_WARNING, &quot;The bitrate parameter is set too low.&quot;

                                             &quot;It takes bits/s as argument, not kbits/s\n&quot;);

            extra_size += ost-&gt;st-&gt;codec-&gt;extradata_size;

        }

    }



    /* open each decoder */

    for (i = 0; i &lt; nb_input_streams; i++) {

        ist = &amp;input_streams[i];

        if (ist-&gt;decoding_needed) {

            AVCodec *codec = ist-&gt;dec;

            if (!codec)

                codec = avcodec_find_decoder(ist-&gt;st-&gt;codec-&gt;codec_id);

            if (!codec) {

                snprintf(error, sizeof(error), &quot;Decoder (codec id %d) not found for input stream #%d.%d&quot;,

                        ist-&gt;st-&gt;codec-&gt;codec_id, ist-&gt;file_index, ist-&gt;st-&gt;index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            if (avcodec_open2(ist-&gt;st-&gt;codec, codec, &amp;ist-&gt;opts) &lt; 0) {

                snprintf(error, sizeof(error), &quot;Error while opening decoder for input stream #%d.%d&quot;,

                        ist-&gt;file_index, ist-&gt;st-&gt;index);

                ret = AVERROR(EINVAL);

                goto dump_format;

            }

            assert_codec_experimental(ist-&gt;st-&gt;codec, 0);

            assert_avoptions(ost-&gt;opts);

            //if (ist-&gt;st-&gt;codec-&gt;codec_type == AVMEDIA_TYPE_VIDEO)

            //    ist-&gt;st-&gt;codec-&gt;flags |= CODEC_FLAG_REPEAT_FIELD;

        }

    }



    /* init pts */

    for (i = 0; i &lt; nb_input_streams; i++) {

        AVStream *st;

        ist = &amp;input_streams[i];

        st= ist-&gt;st;

        ist-&gt;pts = st-&gt;avg_frame_rate.num ? - st-&gt;codec-&gt;has_b_frames*AV_TIME_BASE / av_q2d(st-&gt;avg_frame_rate) : 0;

        ist-&gt;next_pts = AV_NOPTS_VALUE;

        ist-&gt;is_start = 1;

    }



    /* set meta data information from input file if required */

    for (i=0;i&lt;nb_meta_data_maps;i++) {

        AVFormatContext *files[2];

        AVDictionary    **meta[2];

        int j;



#define METADATA_CHECK_INDEX(index, nb_elems, desc)\

        if ((index) &lt; 0 || (index) &gt;= (nb_elems)) {\

            snprintf(error, sizeof(error), &quot;Invalid %s index %d while processing metadata maps\n&quot;,\

                     (desc), (index));\

            ret = AVERROR(EINVAL);\

            goto dump_format;\

        }



        int out_file_index = meta_data_maps[i][0].file;

        int in_file_index = meta_data_maps[i][1].file;

        if (in_file_index &lt; 0 || out_file_index &lt; 0)

            continue;

        METADATA_CHECK_INDEX(out_file_index, nb_output_files, &quot;output file&quot;)

        METADATA_CHECK_INDEX(in_file_index, nb_input_files, &quot;input file&quot;)



        files[0] = output_files[out_file_index];

        files[1] = input_files[in_file_index].ctx;



        for (j = 0; j &lt; 2; j++) {

            MetadataMap *map = &amp;meta_data_maps[i][j];



            switch (map-&gt;type) {

            case &#x27;g&#x27;:

                meta[j] = &amp;files[j]-&gt;metadata;

                break;

            case &#x27;s&#x27;:

                METADATA_CHECK_INDEX(map-&gt;index, files[j]-&gt;nb_streams, &quot;stream&quot;)

                meta[j] = &amp;files[j]-&gt;streams[map-&gt;index]-&gt;metadata;

                break;

            case &#x27;c&#x27;:

                METADATA_CHECK_INDEX(map-&gt;index, files[j]-&gt;nb_chapters, &quot;chapter&quot;)

                meta[j] = &amp;files[j]-&gt;chapters[map-&gt;index]-&gt;metadata;

                break;

            case &#x27;p&#x27;:

                METADATA_CHECK_INDEX(map-&gt;index, files[j]-&gt;nb_programs, &quot;program&quot;)

                meta[j] = &amp;files[j]-&gt;programs[map-&gt;index]-&gt;metadata;

                break;

            }

        }



        av_dict_copy(meta[0], *meta[1], AV_DICT_DONT_OVERWRITE);

    }



    /* copy global metadata by default */

    if (metadata_global_autocopy) {



        for (i = 0; i &lt; nb_output_files; i++)

            av_dict_copy(&amp;output_files[i]-&gt;metadata, input_files[0].ctx-&gt;metadata,

                         AV_DICT_DONT_OVERWRITE);

    }



    /* copy chapters according to chapter maps */

    for (i = 0; i &lt; nb_chapter_maps; i++) {

        int infile  = chapter_maps[i].in_file;

        int outfile = chapter_maps[i].out_file;



        if (infile &lt; 0 || outfile &lt; 0)

            continue;

        if (infile &gt;= nb_input_files) {

            snprintf(error, sizeof(error), &quot;Invalid input file index %d in chapter mapping.\n&quot;, infile);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        if (outfile &gt;= nb_output_files) {

            snprintf(error, sizeof(error), &quot;Invalid output file index %d in chapter mapping.\n&quot;,outfile);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        copy_chapters(infile, outfile);

    }



    /* copy chapters from the first input file that has them*/

    if (!nb_chapter_maps)

        for (i = 0; i &lt; nb_input_files; i++) {

            if (!input_files[i].ctx-&gt;nb_chapters)

                continue;



            for (j = 0; j &lt; nb_output_files; j++)

                if ((ret = copy_chapters(i, j)) &lt; 0)

                    goto dump_format;

            break;

        }



    /* open files and write file headers */

    for(i=0;i&lt;nb_output_files;i++) {

        os = output_files[i];

        if (avformat_write_header(os, &amp;output_opts[i]) &lt; 0) {

            snprintf(error, sizeof(error), &quot;Could not write header for output file #%d (incorrect codec parameters ?)&quot;, i);

            ret = AVERROR(EINVAL);

            goto dump_format;

        }

        assert_avoptions(output_opts[i]);

        if (strcmp(output_files[i]-&gt;oformat-&gt;name, &quot;rtp&quot;)) {

            want_sdp = 0;

        }

    }



 dump_format:

    /* dump the file output parameters - cannot be done before in case

       of stream copy */

    for(i=0;i&lt;nb_output_files;i++) {

        av_dump_format(output_files[i], i, output_files[i]-&gt;filename, 1);

    }



    /* dump the stream mapping */

    if (verbose &gt;= 0) {

        fprintf(stderr, &quot;Stream mapping:\n&quot;);

        for(i=0;i&lt;nb_ostreams;i++) {

            ost = ost_table[i];

            fprintf(stderr, &quot;  Stream #%d.%d -&gt; #%d.%d&quot;,

                    input_streams[ost-&gt;source_index].file_index,

                    input_streams[ost-&gt;source_index].st-&gt;index,

                    ost-&gt;file_index,

                    ost-&gt;index);

            if (ost-&gt;sync_ist != &amp;input_streams[ost-&gt;source_index])

                fprintf(stderr, &quot; [sync #%d.%d]&quot;,

                        ost-&gt;sync_ist-&gt;file_index,

                        ost-&gt;sync_ist-&gt;st-&gt;index);

            fprintf(stderr, &quot;\n&quot;);

        }

    }



    if (ret) {

        fprintf(stderr, &quot;%s\n&quot;, error);

        goto fail;

    }



    if (want_sdp) {

        print_sdp(output_files, nb_output_files);

    }



    if (!using_stdin) {

        if(verbose &gt;= 0)

            fprintf(stderr, &quot;Press [q] to stop, [?] for help\n&quot;);

        avio_set_interrupt_cb(decode_interrupt_cb);

    }

    term_init();



    timer_start = av_gettime();



    for(; received_sigterm == 0;) {

        int file_index, ist_index;

        AVPacket pkt;

        double ipts_min;

        double opts_min;



    redo:

        ipts_min= 1e100;

        opts_min= 1e100;

        /* if &#x27;q&#x27; pressed, exits */

        if (!using_stdin) {

            if (q_pressed)

                break;

            /* read_key() returns 0 on EOF */

            key = read_key();

            if (key == &#x27;q&#x27;)

                break;

            if (key == &#x27;+&#x27;) verbose++;

            if (key == &#x27;-&#x27;) verbose--;

            if (key == &#x27;s&#x27;) qp_hist     ^= 1;

            if (key == &#x27;h&#x27;){

                if (do_hex_dump){

                    do_hex_dump = do_pkt_dump = 0;

                } else if(do_pkt_dump){

                    do_hex_dump = 1;

                } else

                    do_pkt_dump = 1;

                av_log_set_level(AV_LOG_DEBUG);

            }

            if (key == &#x27;d&#x27; || key == &#x27;D&#x27;){

                int debug=0;

                if(key == &#x27;D&#x27;) {

                    debug = input_streams[0].st-&gt;codec-&gt;debug&lt;&lt;1;

                    if(!debug) debug = 1;

                    while(debug &amp; (FF_DEBUG_DCT_COEFF|FF_DEBUG_VIS_QP|FF_DEBUG_VIS_MB_TYPE)) //unsupported, would just crash

                        debug += debug;

                }else

                    scanf(&quot;%d&quot;, &amp;debug);

                for(i=0;i&lt;nb_input_streams;i++) {

                    input_streams[i].st-&gt;codec-&gt;debug = debug;

                }

                for(i=0;i&lt;nb_ostreams;i++) {

                    ost = ost_table[i];

                    ost-&gt;st-&gt;codec-&gt;debug = debug;

                }

                if(debug) av_log_set_level(AV_LOG_DEBUG);

                fprintf(stderr,&quot;debug=%d\n&quot;, debug);

            }

            if (key == &#x27;?&#x27;){

                fprintf(stderr, &quot;key    function\n&quot;

                                &quot;?      show this help\n&quot;

                                &quot;+      increase verbosity\n&quot;

                                &quot;-      decrease verbosity\n&quot;

                                &quot;D      cycle through available debug modes\n&quot;

                                &quot;h      dump packets/hex press to cycle through the 3 states\n&quot;

                                &quot;q      quit\n&quot;

                                &quot;s      Show QP histogram\n&quot;

                );

            }

        }



        /* select the stream that we must read now by looking at the

           smallest output pts */

        file_index = -1;

        for(i=0;i&lt;nb_ostreams;i++) {

            double ipts, opts;

            ost = ost_table[i];

            os = output_files[ost-&gt;file_index];

            ist = &amp;input_streams[ost-&gt;source_index];

            if(ist-&gt;is_past_recording_time || no_packet[ist-&gt;file_index])

                continue;

                opts = ost-&gt;st-&gt;pts.val * av_q2d(ost-&gt;st-&gt;time_base);

            ipts = (double)ist-&gt;pts;

            if (!input_files[ist-&gt;file_index].eof_reached){

                if(ipts &lt; ipts_min) {

                    ipts_min = ipts;

                    if(input_sync ) file_index = ist-&gt;file_index;

                }

                if(opts &lt; opts_min) {

                    opts_min = opts;

                    if(!input_sync) file_index = ist-&gt;file_index;

                }

            }

            if(ost-&gt;frame_number &gt;= max_frames[ost-&gt;st-&gt;codec-&gt;codec_type]){

                file_index= -1;

                break;

            }

        }

        /* if none, if is finished */

        if (file_index &lt; 0) {

            if(no_packet_count){

                no_packet_count=0;

                memset(no_packet, 0, sizeof(no_packet));

                usleep(10000);

                continue;

            }

            break;

        }



        /* finish if limit size exhausted */

        if (limit_filesize != 0 &amp;&amp; limit_filesize &lt;= avio_tell(output_files[0]-&gt;pb))

            break;



        /* read a frame from it and output it in the fifo */

        is = input_files[file_index].ctx;

        ret= av_read_frame(is, &amp;pkt);

        if(ret == AVERROR(EAGAIN)){

            no_packet[file_index]=1;

            no_packet_count++;

            continue;

        }

        if (ret &lt; 0) {

            input_files[file_index].eof_reached = 1;

            if (opt_shortest)

                break;

            else

                continue;

        }



        no_packet_count=0;

        memset(no_packet, 0, sizeof(no_packet));



        if (do_pkt_dump) {

            av_pkt_dump_log2(NULL, AV_LOG_DEBUG, &amp;pkt, do_hex_dump,

                             is-&gt;streams[pkt.stream_index]);

        }

        /* the following test is needed in case new streams appear

           dynamically in stream : we ignore them */

        if (pkt.stream_index &gt;= input_files[file_index].ctx-&gt;nb_streams)

            goto discard_packet;

        ist_index = input_files[file_index].ist_index + pkt.stream_index;

        ist = &amp;input_streams[ist_index];

        if (ist-&gt;discard)

            goto discard_packet;



        if (pkt.dts != AV_NOPTS_VALUE)

            pkt.dts += av_rescale_q(input_files[ist-&gt;file_index].ts_offset, AV_TIME_BASE_Q, ist-&gt;st-&gt;time_base);

        if (pkt.pts != AV_NOPTS_VALUE)

            pkt.pts += av_rescale_q(input_files[ist-&gt;file_index].ts_offset, AV_TIME_BASE_Q, ist-&gt;st-&gt;time_base);



        if (ist-&gt;ts_scale) {

            if(pkt.pts != AV_NOPTS_VALUE)

                pkt.pts *= ist-&gt;ts_scale;

            if(pkt.dts != AV_NOPTS_VALUE)

                pkt.dts *= ist-&gt;ts_scale;

        }



//        fprintf(stderr, &quot;next:%&quot;PRId64&quot; dts:%&quot;PRId64&quot; off:%&quot;PRId64&quot; %d\n&quot;, ist-&gt;next_pts, pkt.dts, input_files[ist-&gt;file_index].ts_offset, ist-&gt;st-&gt;codec-&gt;codec_type);

        if (pkt.dts != AV_NOPTS_VALUE &amp;&amp; ist-&gt;next_pts != AV_NOPTS_VALUE

            &amp;&amp; (is-&gt;iformat-&gt;flags &amp; AVFMT_TS_DISCONT)) {

            int64_t pkt_dts= av_rescale_q(pkt.dts, ist-&gt;st-&gt;time_base, AV_TIME_BASE_Q);

            int64_t delta= pkt_dts - ist-&gt;next_pts;

            if((FFABS(delta) &gt; 1LL*dts_delta_threshold*AV_TIME_BASE || pkt_dts+1&lt;ist-&gt;pts)&amp;&amp; !copy_ts){

                input_files[ist-&gt;file_index].ts_offset -= delta;

                if (verbose &gt; 2)

                    fprintf(stderr, &quot;timestamp discontinuity %&quot;PRId64&quot;, new offset= %&quot;PRId64&quot;\n&quot;,

                            delta, input_files[ist-&gt;file_index].ts_offset);

                pkt.dts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist-&gt;st-&gt;time_base);

                if(pkt.pts != AV_NOPTS_VALUE)

                    pkt.pts-= av_rescale_q(delta, AV_TIME_BASE_Q, ist-&gt;st-&gt;time_base);

            }

        }



        /* finish if recording time exhausted */

        if (recording_time != INT64_MAX &amp;&amp;

            (pkt.pts != AV_NOPTS_VALUE ?

                av_compare_ts(pkt.pts, ist-&gt;st-&gt;time_base, recording_time + start_time, (AVRational){1, 1000000})

                    :

                av_compare_ts(ist-&gt;pts, AV_TIME_BASE_Q, recording_time + start_time, (AVRational){1, 1000000})

            )&gt;= 0) {

            ist-&gt;is_past_recording_time = 1;

            goto discard_packet;

        }



        //fprintf(stderr,&quot;read #%d.%d size=%d\n&quot;, ist-&gt;file_index, ist-&gt;st-&gt;index, pkt.size);

        if (output_packet(ist, ist_index, ost_table, nb_ostreams, &amp;pkt) &lt; 0) {



            if (verbose &gt;= 0)

                fprintf(stderr, &quot;Error while decoding stream #%d.%d\n&quot;,

                        ist-&gt;file_index, ist-&gt;st-&gt;index);

            if (exit_on_error)

                ffmpeg_exit(1);

            av_free_packet(&amp;pkt);

            goto redo;

        }



    discard_packet:

        av_free_packet(&amp;pkt);



        /* dump report by using the output first video and audio streams */

        print_report(output_files, ost_table, nb_ostreams, 0);

    }



    /* at the end of stream, we must flush the decoder buffers */

    for (i = 0; i &lt; nb_input_streams; i++) {

        ist = &amp;input_streams[i];

        if (ist-&gt;decoding_needed) {

            output_packet(ist, i, ost_table, nb_ostreams, NULL);

        }

    }



    term_exit();



    /* write the trailer if needed and close file */

    for(i=0;i&lt;nb_output_files;i++) {

        os = output_files[i];

        av_write_trailer(os);

    }



    /* dump report by using the first video and audio streams */

    print_report(output_files, ost_table, nb_ostreams, 1);



    /* close each encoder */

    for(i=0;i&lt;nb_ostreams;i++) {

        ost = ost_table[i];

        if (ost-&gt;encoding_needed) {

            av_freep(&amp;ost-&gt;st-&gt;codec-&gt;stats_in);

            avcodec_close(ost-&gt;st-&gt;codec);

        }

#if CONFIG_AVFILTER

        avfilter_graph_free(&amp;ost-&gt;graph);

#endif

    }



    /* close each decoder */

    for (i = 0; i &lt; nb_input_streams; i++) {

        ist = &amp;input_streams[i];

        if (ist-&gt;decoding_needed) {

            avcodec_close(ist-&gt;st-&gt;codec);

        }

    }



    /* finished ! */

    ret = 0;



 fail:

    av_freep(&amp;bit_buffer);



    if (ost_table) {

        for(i=0;i&lt;nb_ostreams;i++) {

            ost = ost_table[i];

            if (ost) {

                if (ost-&gt;st-&gt;stream_copy)

                    av_freep(&amp;ost-&gt;st-&gt;codec-&gt;extradata);

                if (ost-&gt;logfile) {

                    fclose(ost-&gt;logfile);

                    ost-&gt;logfile = NULL;

                }

                av_fifo_free(ost-&gt;fifo); /* works even if fifo is not

                                             initialized but set to zero */

                av_freep(&amp;ost-&gt;st-&gt;codec-&gt;subtitle_header);

                av_free(ost-&gt;resample_frame.data[0]);

                av_free(ost-&gt;forced_kf_pts);

                if (ost-&gt;video_resample)

                    sws_freeContext(ost-&gt;img_resample_ctx);

                if (ost-&gt;resample)

                    audio_resample_close(ost-&gt;resample);

                if (ost-&gt;reformat_ctx)

                    av_audio_convert_free(ost-&gt;reformat_ctx);

                av_dict_free(&amp;ost-&gt;opts);

                av_free(ost);

            }

        }

        av_free(ost_table);

    }

    return ret;

}
</pre>

**Question**

<pre>Does this C function contain a defect, such as a vulnerability or a memory bug?</pre>

**Criteria (choice)**

<pre>1. defect
2. no defect</pre>

**Gold target (weight)**

<pre>2. no defect — 1</pre>

</details>

<details><summary>Example 2</summary>

published Jev · train · code-x-glue-cc-defect-detection-1c76c57e72:train:19865

**State**

<pre>static uint32_t openpic_cpu_read_internal(void *opaque, hwaddr addr,

                                          int idx)

{

    openpic_t *opp = opaque;

    IRQ_src_t *src;

    IRQ_dst_t *dst;

    uint32_t retval;

    int n_IRQ;



    DPRINTF(&quot;%s: cpu %d addr &quot; TARGET_FMT_plx &quot;\n&quot;, __func__, idx, addr);

    retval = 0xFFFFFFFF;

    if (addr &amp; 0xF)

        return retval;

    dst = &amp;opp-&gt;dst[idx];

    addr &amp;= 0xFF0;

    switch (addr) {

    case 0x00: /* Block Revision Register1 (BRR1) */

        retval = FSL_BRR1_IPID | FSL_BRR1_IPMJ | FSL_BRR1_IPMN;

        break;

    case 0x80: /* PCTP */

        retval = dst-&gt;pctp;

        break;

    case 0x90: /* WHOAMI */

        retval = idx;

        break;

    case 0xA0: /* PIAC */

        DPRINTF(&quot;Lower OpenPIC INT output\n&quot;);

        qemu_irq_lower(dst-&gt;irqs[OPENPIC_OUTPUT_INT]);

        n_IRQ = IRQ_get_next(opp, &amp;dst-&gt;raised);

        DPRINTF(&quot;PIAC: irq=%d\n&quot;, n_IRQ);

        if (n_IRQ == -1) {

            /* No more interrupt pending */

            retval = IPVP_VECTOR(opp-&gt;spve);

        } else {

            src = &amp;opp-&gt;src[n_IRQ];

            if (!test_bit(&amp;src-&gt;ipvp, IPVP_ACTIVITY) ||

                !(IPVP_PRIORITY(src-&gt;ipvp) &gt; dst-&gt;pctp)) {

                /* - Spurious level-sensitive IRQ

                 * - Priorities has been changed

                 *   and the pending IRQ isn&#x27;t allowed anymore

                 */

                reset_bit(&amp;src-&gt;ipvp, IPVP_ACTIVITY);

                retval = IPVP_VECTOR(opp-&gt;spve);

            } else {

                /* IRQ enter servicing state */

                IRQ_setbit(&amp;dst-&gt;servicing, n_IRQ);

                retval = IPVP_VECTOR(src-&gt;ipvp);

            }

            IRQ_resetbit(&amp;dst-&gt;raised, n_IRQ);

            dst-&gt;raised.next = -1;

            if (!test_bit(&amp;src-&gt;ipvp, IPVP_SENSE)) {

                /* edge-sensitive IRQ */

                reset_bit(&amp;src-&gt;ipvp, IPVP_ACTIVITY);

                src-&gt;pending = 0;

            }



            if ((n_IRQ &gt;= opp-&gt;irq_ipi0) &amp;&amp;  (n_IRQ &lt; (opp-&gt;irq_ipi0 + MAX_IPI))) {

                src-&gt;ide &amp;= ~(1 &lt;&lt; idx);

                if (src-&gt;ide &amp;&amp; !test_bit(&amp;src-&gt;ipvp, IPVP_SENSE)) {

                    /* trigger on CPUs that didn&#x27;t know about it yet */

                    openpic_set_irq(opp, n_IRQ, 1);

                    openpic_set_irq(opp, n_IRQ, 0);

                    /* if all CPUs knew about it, set active bit again */

                    set_bit(&amp;src-&gt;ipvp, IPVP_ACTIVITY);

                }

            }

        }

        break;

    case 0xB0: /* PEOI */

        retval = 0;

        break;

    default:

        break;

    }

    DPRINTF(&quot;%s: =&gt; %08x\n&quot;, __func__, retval);



    return retval;

}
</pre>

**Question**

<pre>Does this C function contain a defect, such as a vulnerability or a memory bug?</pre>

**Criteria (choice)**

<pre>1. defect
2. no defect</pre>

**Gold target (weight)**

<pre>2. no defect — 1</pre>

</details>

<details><summary>Example 3</summary>

published Jev · train · code-x-glue-cc-defect-detection-1c76c57e72:train:8239

**State**

<pre>static int xen_pt_msixctrl_reg_init(XenPCIPassthroughState *s,

                                    XenPTRegInfo *reg, uint32_t real_offset,

                                    uint32_t *data)

{

    PCIDevice *d = &amp;s-&gt;dev;

    uint16_t reg_field = 0;



    /* use I/O device register&#x27;s value as initial value */

    reg_field = pci_get_word(d-&gt;config + real_offset);



    if (reg_field &amp; PCI_MSIX_FLAGS_ENABLE) {

        XEN_PT_LOG(d, &quot;MSIX already enabled, disabling it first\n&quot;);

        xen_host_pci_set_word(&amp;s-&gt;real_device, real_offset,

                              reg_field &amp; ~PCI_MSIX_FLAGS_ENABLE);

    }



    s-&gt;msix-&gt;ctrl_offset = real_offset;



    *data = reg-&gt;init_val;

    return 0;

}
</pre>

**Question**

<pre>Does this C function contain a defect, such as a vulnerability or a memory bug?</pre>

**Criteria (choice)**

<pre>1. defect
2. no defect</pre>

**Gold target (weight)**

<pre>2. no defect — 1</pre>

</details>
