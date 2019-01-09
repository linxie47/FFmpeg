/*
 * IE meta data muxer
 * Copyright (c) 2019 Shaofei Wang
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#include "avformat.h"
#include "internal.h"
#include "libavutil/dict.h"
#include "libavfilter/avfilter.h"
#include "libavfilter/inference.h"
#include "libavutil/opt.h"
#include <float.h>

#define JSON_HEAD "{\n"
#define JSON_TAIL "},\n"
#define JSON_ESCAPE "    "
#define JSON_ARRAY_TAIL "]\n"
#define JSON_FVALUE(str, name, value) snprintf(str, TMP_STR_BUF_SIZE, "\"%s\": %.1f,\n", name, value)
#define JSON_IVALUE(str, name, value) snprintf(str, TMP_STR_BUF_SIZE, "\"%s\": %d,\n", name, value)
#define JSON_LIVALUE(str, name, value) snprintf(str,TMP_STR_BUF_SIZE, "\"%s\": %lu,\n", name, value)
#define JSON_STRING(str, name, value) snprintf(str, TMP_STR_BUF_SIZE, "\"%s\": \"%s\",\n", name, value)

#define BUFFER_SIZE (1024 * 1024)
#define TMP_STR_BUF_SIZE 4096

typedef struct IeMetaDataMuxer {
    const AVClass *class;
    char *meta_data_strings;
    size_t meta_data_length;
    unsigned int current_escape_num;
    char *source;
    char *tag;
    int id_number;
    int output_type;
} IeMetaDataMuxer;

static int fill_content(AVFormatContext *s, const char *str, int flush)
{
    IeMetaDataMuxer *md = s->priv_data;
    unsigned int len;

    if (!str)
        return 0;

    len = strlen(str);
    if (str[len] == '\0')
        len++;
    if (((len + md->meta_data_length) > BUFFER_SIZE)) {
        avio_write(s->pb, md->meta_data_strings, md->meta_data_length);
        avio_flush(s->pb);
        md->meta_data_length = 0;
    }
    memcpy(md->meta_data_strings + md->meta_data_length, str, len);
    md->meta_data_length += len - 1;
    if (flush) {
        avio_write(s->pb, md->meta_data_strings, md->meta_data_length);
        avio_flush(s->pb);
        md->meta_data_length = 0;
    }
    return md->meta_data_length;
}

static int escape(AVFormatContext *s, unsigned int n)
{
    unsigned int i;
    for (i = 0; i < n; i++)
        fill_content(s, JSON_ESCAPE, 0);
    return 0;
}

static int fill_line(AVFormatContext *s, const char *str, unsigned int num_esp,  int flush)
{
    escape(s, num_esp);
    fill_content(s, str, flush);
    return 0;
}

static int pack(AVFormatContext *s, const char *org, ...)
{
    va_list argp;
    int i, len;
    char tmp_str[TMP_STR_BUF_SIZE];
    const char *p = org;
    char *name, *str;
    int ipara;
    int64_t lipara;
    float fpara;
    IeMetaDataMuxer *md = s->priv_data;
    len = strlen(org);

    va_start(argp, org);
    for (i = 0; i < len; i++) {
        switch (p[i]) {
            case '{':
                fill_line(s, JSON_HEAD, md->current_escape_num, 0);
                ++md->current_escape_num;
                break;
            case '}':
            case ')':
            case ']':
                if (md->meta_data_strings[md->meta_data_length - 2] == ',') {
                    md->meta_data_strings[md->meta_data_length - 2] =
                        md->meta_data_strings[md->meta_data_length - 1];
                    --md->meta_data_length;
                }
                escape(s, --md->current_escape_num);
                if (p[i] != ']')
                    fill_content(s, "}", 0);
                else
                    fill_content(s, "]", 0);
                break;
            case '[': //group head
                str = va_arg(argp, char*);
                snprintf(tmp_str, TMP_STR_BUF_SIZE, "\"%s\": [\n", str);
                fill_line(s, tmp_str, md->current_escape_num, 0);
                ++md->current_escape_num;
                break;
            case '(': //element head
                str = va_arg(argp, char*);
                snprintf(tmp_str, TMP_STR_BUF_SIZE, "\"%s\": {\n", str);
                fill_line(s, tmp_str, md->current_escape_num, 0);
                ++md->current_escape_num;
                break;
            case 'i': //int value
            case 'I': //long int value
            case 'f': //float value
            case 's': //str value
                name = va_arg(argp, char*);
                if (p[i] == 'i') {
                    ipara = va_arg(argp, int);
                    JSON_IVALUE(tmp_str, name, ipara);
                } else if (p[i] == 'I') {
                    lipara = va_arg(argp, int64_t);
                    JSON_LIVALUE(tmp_str, name, lipara);
                } else if (p[i] == 'f') {
                    fpara = va_arg(argp, double);
                    JSON_FVALUE(tmp_str, name, fpara);
                } else {
                    str = va_arg(argp, char *);
                    JSON_STRING(tmp_str, name, str);
                }
                fill_line(s, tmp_str, md->current_escape_num, 0);
                break;
            case 'S': //str only
                str = va_arg(argp, char *);
                fill_line(s, str, md->current_escape_num, 0);
                break;
            case ':':
                fill_content(s, ": ", 0);
                break;
            case ',':
                fill_content(s, ",\n", 0);
                break;
            case 'n': // \n
                fill_content(s, "\n", 0);
                break;
            case 'w': //flush write
                avio_write(s->pb, md->meta_data_strings, md->meta_data_length);
                avio_flush(s->pb);
                md->meta_data_length = 0;
                break;
            default:
                break;
        }
    }
    return 0;
}

static int init(AVFormatContext *s)
{
    IeMetaDataMuxer *md = s->priv_data;

    md->meta_data_strings = av_mallocz(BUFFER_SIZE);
    if (!md->meta_data_strings) {
        av_log(s, AV_LOG_ERROR, "fail to alloc buffer for meta data\n");
    }
    md->meta_data_length = 0;
    md->current_escape_num = 0;
    md->id_number = 0;

    if (md->output_type == 1)
        pack(s, "{");

    return 0;
}

static void deinit(AVFormatContext *s)
{
    IeMetaDataMuxer *md = s->priv_data;
    if (md->output_type == 1) {
        pack(s, "n}w");
    }
    av_free(md->meta_data_strings);

}

static int write_header(AVFormatContext *s)
{
    //fill_content(s, JSON_HEAD, sizeof(JSON_HEAD), 1);
    return 0;
}

static int write_trailer(AVFormatContext *s)
{
    //fill_content(s, JSON_TAIL, sizeof(JSON_TAIL), 1);
    return 0;
}

static int jhead_write(AVFormatContext *s, AVFrame *frm_data)
{
    char tmp_str[TMP_STR_BUF_SIZE];
    IeMetaDataMuxer *md = s->priv_data;
    int64_t nano_ts = 1000000000;

    if (s->streams[0])
        nano_ts = frm_data->pts * (nano_ts * s->streams[0]->time_base.num / s->streams[0]->time_base.den);
    else
        nano_ts = -1;

    memset(tmp_str, 0, TMP_STR_BUF_SIZE);

    if (md->output_type == 1 && md->id_number != 0) {
        pack(s, ",");
    }

    snprintf(tmp_str, TMP_STR_BUF_SIZE, "\"resolution\":{\"width\":%d,\"height\":%d},\n",
            frm_data->width, frm_data->height);
    if (md->output_type == 0)
        pack(s, "{IsS", "timestamp", nano_ts,
                "source", md->source,
                tmp_str);
    else {
        char id[80];
        snprintf(id, 80, "id-%d", md->id_number++);
        pack(s, "(IsS", id, "timestamp", nano_ts,
                "source", md->source,
                tmp_str);
    }

    if (!md->tag)
        snprintf(tmp_str, TMP_STR_BUF_SIZE, "\"tags\":{\"custom_key\":\"custom_value\"},\n");
    else {
        char *token, *save_ptr, *tag_str;
        int offset;
        char key[128] = "";
        char tags[256];
        float value = 0.0;
        int len = (strlen(md->tag) < 255) ? strlen(md->tag) : 255;

        memset(tags, 0, 256);
        memcpy(tags, md->tag, len);
        offset = snprintf(tmp_str, TMP_STR_BUF_SIZE, "\"tags\":{");
        for (tag_str = tags; ; tag_str = NULL) {
            token = strtok_r(tag_str, ",", &save_ptr);
            if (token == NULL)
                break;
            sscanf(token, "%127[^:]:%f", key, &value);
            offset += snprintf(tmp_str + offset, TMP_STR_BUF_SIZE - offset, "\"%s\":%1.3f,", key, value);
        }
        snprintf(tmp_str + offset - 2, TMP_STR_BUF_SIZE - offset + 2, "},\n");
    }
    pack(s, "S[", tmp_str, "objects");

    return 0;
}

static int jtail_write(AVFormatContext *s)
{
    IeMetaDataMuxer *md = s->priv_data;

    if (md->output_type == 0)
        pack(s, "]n}nw");
    else
        pack(s, "]n}w");
    md->meta_data_length = 0;

    return 0;
}

static int write_packet(AVFormatContext *s, AVPacket *pkt)
{
    int i, j, head_written = 0;
    char tmp_str[TMP_STR_BUF_SIZE];
    AVFrame *frm_data = (AVFrame *)pkt->data;
    AVFrameSideData *sd;
    InferDetectionMeta *meta;
    BBoxesArray *bboxes;
    AVFrameSideData *c_sd;
    InferClassificationMeta *cmeta;
    ClassifyArray *c_array;

    if (!frm_data)
        return 0;

    sd = av_frame_get_side_data(frm_data, AV_FRAME_DATA_INFERENCE_DETECTION);
    c_sd = av_frame_get_side_data(frm_data, AV_FRAME_DATA_INFERENCE_CLASSIFICATION);
    if (sd) {
        meta = (InferDetectionMeta *)sd->data;

        if (meta) {
            bboxes = meta->bboxes;
            if (bboxes) {
                if (bboxes->num > 0) {
                    jhead_write(s, frm_data);
                    head_written = 1;
                }

                for (i = 0; i < bboxes->num; i++) {
                    if (!bboxes->bbox[i]->label_buf) {
                        snprintf(tmp_str, TMP_STR_BUF_SIZE, "%s", "face");
                    } else {
                        int label_id = bboxes->bbox[i]->label_id;
                        LabelsArray *array = (LabelsArray*)(bboxes->bbox[i]->label_buf->data);
                        snprintf(tmp_str, TMP_STR_BUF_SIZE, "%s", array->label[label_id]);
                    }

                    pack(s, "{((ffff),isifS),",
                            "detection", "bounding_box",
                            "x_min", bboxes->bbox[i]->x_min,
                            "y_min", bboxes->bbox[i]->y_min,
                            "x_max", bboxes->bbox[i]->x_max,
                            "y_max", bboxes->bbox[i]->y_max,
                            "object_id", bboxes->bbox[i]->object_id,
                            "label", tmp_str,
                            "label_id", bboxes->bbox[i]->label_id,
                            "confidence", bboxes->bbox[i]->confidence,
                            "\"model\":{\"name\":\"\", \"version\":1},\n");

                    //emotion, age, gender
                    if (c_sd) {
                        cmeta = (InferClassificationMeta *)c_sd->data;
                        if (cmeta) {
                            c_array = cmeta->c_array;
                            if (c_array) {
                                for (j = 0; j < c_array->num; j++) {
                                    if (c_array->classifications[j]->detect_id == i) {
                                        char *name = c_array->classifications[j]->name;
                                        if (strncmp(name, "emotion", strlen("emotion")) == 0 ||
                                                strncmp(name, "gender", strlen("gender")) == 0 ||
                                                strncmp(name, "face_id", strlen("face_id")) == 0) {
                                            pack(s, "(sifS),",
                                                    name,
                                                    "label", ((LabelsArray*)c_array->classifications[j]->label_buf->data)->label[c_array->classifications[j]->label_id],
                                                    "label_id", c_array->classifications[j]->label_id,
                                                    "confidence", c_array->classifications[j]->confidence,
                                                    "\"model\":{\"name\":\"\", \"version\":1},\n");
                                        } else if (strncmp(name, "age", strlen("age")) == 0) {
                                            pack(s, "(ffS),",
                                                    name,
                                                    "value", c_array->classifications[j]->value,
                                                    "confidence", c_array->classifications[j]->confidence,
                                                    "\"model\":{\"name\":\"\", \"version\":1},\n");
                                        }
                                    }
                                }
                            }
                        }
                    }
                    pack(s, "},");
                }
            }
        }
    }
    if (head_written)
        jtail_write(s);

    return 0;
}

#define OFFSET(x) offsetof(IeMetaDataMuxer, x)
#define ENC AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    { "source_url", "the source url/path to put into the json metadata", OFFSET(source), AV_OPT_TYPE_STRING, { .str = "auto" }, 0, 0, ENC },
    { "custom_tag", "the customer tag and value, usage: -custom_tag \"key1:value1,key2:value2\"", OFFSET(tag), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, ENC },
    { "output_type", "it will output meta data frame by frame by default 0, otherwise 1 means file output which group all the data. usage: -output_type 1\"", OFFSET(output_type), AV_OPT_TYPE_INT, {.i64 = 0}, 0, 1,ENC },
    { NULL },
};

static const AVClass iemetadata_muxer_class = {
    .class_name = "iemetadata muxer",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

AVOutputFormat ff_iemetadata_muxer = {
    .name           = "iemetadata",
    .long_name      = NULL_IF_CONFIG_SMALL("Inference engine meta data muxer"),
    .extensions     = "json",
    .priv_data_size = sizeof(IeMetaDataMuxer),
    .priv_class     = &iemetadata_muxer_class,
    .init           = init,
    .deinit         = deinit,
    .video_codec    = AV_CODEC_ID_WRAPPED_AVFRAME,
    .write_header   = write_header,
    .write_packet   = write_packet,
    .write_trailer  = write_trailer,
    .flags          = AVFMT_VARIABLE_FPS,
};
