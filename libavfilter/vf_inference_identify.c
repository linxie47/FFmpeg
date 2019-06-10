/*
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

/**
 * @file
 * dnn inference identify filter
 */
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/eval.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/pixdesc.h"
#include "libavformat/avformat.h"

#include "formats.h"
#include "internal.h"
#include "avfilter.h"

#include "inference.h"

#include <cjson/cJSON.h>

#define OFFSET(x) offsetof(InferenceIdentifyContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

#define PI 3.1415926
#define FACE_FEATURE_VECTOR_LEN 256

typedef struct FeatureLabelPair {
    float *feature;
    size_t label_id;
} FeatureLabelPair;

typedef struct InferenceIdentifyContext {
    const AVClass *class;

    char   *gallery;      ///<< gallery for identify features
    double *norm_std;

    AVBufferRef *labels;
    FeatureLabelPair **features;
    int features_num;
} InferenceIdentifyContext;

static const char *get_filename_ext(const char *filename) {
    const char *dot = strrchr(filename, '.');
    if (!dot || dot == filename)
        return NULL;

    return dot + 1;
}

const char *gallery_file_suffix = "json";

static void infer_labels_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    LabelsArray *labels = (LabelsArray *)data;

    for (i = 0; i < labels->num; i++)
        av_freep(&labels->label[i]);

    av_free(data);
}

static int query_formats(AVFilterContext *context)
{
    AVFilterFormats *formats_list;
    const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV422P,  AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV411P,  AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_BGR24,    AV_PIX_FMT_BGRA,     AV_PIX_FMT_VAAPI,
        AV_PIX_FMT_NONE};

    formats_list = ff_make_format_list(pixel_formats);
    if (!formats_list) {
        av_log(context, AV_LOG_ERROR, "Could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static av_cold int identify_init(AVFilterContext *ctx)
{
    size_t i, index = 1;
    char *dup, *unknown;
    const char *dirname;
    cJSON *entry, *item;
    LabelsArray *larray = NULL;
    AVBufferRef *ref    = NULL;
    InferenceIdentifyContext *s = ctx->priv;
    size_t vec_size_in_bytes = sizeof(float) * FACE_FEATURE_VECTOR_LEN;

    av_assert0(s->gallery);

    if (strcmp(get_filename_ext(s->gallery), gallery_file_suffix)) {
        av_log(ctx, AV_LOG_ERROR, "Face gallery '%s' is not a json file\n", s->gallery);
        return AVERROR(EINVAL);
    }

    entry = ff_read_model_proc(s->gallery);
    if (!entry) {
        av_log(ctx, AV_LOG_ERROR, "Could not open gallery file:%s\n", s->gallery);
        return AVERROR(EIO);
    }

    dup = av_strdup(s->gallery);
    dirname = av_dirname(dup);

    larray = av_mallocz(sizeof(*larray));
    if (!larray)
        return AVERROR(ENOMEM);

    // label id 0 reserved for unknown person
    unknown = av_strdup("Unknown_Person");
    av_dynarray_add(&larray->label, &larray->num, unknown);

    cJSON_ArrayForEach(item, entry)
    {
        char *l = av_strdup(item->string);
        cJSON *features, *feature;

        av_dynarray_add(&larray->label, &larray->num, l);

        features = cJSON_GetObjectItem(item, "features");

        cJSON_ArrayForEach(feature, features)
        {
            FILE *vec_fp;
            FeatureLabelPair *pair;
            char path[4096];

            memset(path, 0, sizeof(path));

            if (!cJSON_IsString(feature) || !feature->valuestring)
                continue;

            strncpy(path, dirname, strlen(dirname));
            strncat(path, "/", 1);
            strncat(path, feature->valuestring, strlen(feature->valuestring));

            vec_fp = fopen(path, "rb");
            if (!vec_fp) {
                av_log(ctx, AV_LOG_ERROR, "Could not open feature file:%s\n", path);
                continue;
            }

            pair = av_mallocz(sizeof(FeatureLabelPair));
            if (!pair)
                return AVERROR(ENOMEM);

            pair->feature = av_malloc(vec_size_in_bytes);
            if (!pair->feature)
                return AVERROR(ENOMEM);

            if (fread(pair->feature, vec_size_in_bytes, 1, vec_fp) != 1) {
                av_log(ctx, AV_LOG_ERROR, "Feature vector size mismatch:%s\n", path);
                fclose(vec_fp);
                return AVERROR(EINVAL);
            }

            fclose(vec_fp);

            pair->label_id = index;
            av_dynarray_add(&s->features, &s->features_num, pair);
        }
        index++;
    }

    s->norm_std = av_mallocz(sizeof(double) * s->features_num);
    if (!s->norm_std)
        return AVERROR(ENOMEM);

    for (i = 0; i < s->features_num; i++)
        s->norm_std[i] = av_norm(s->features[i]->feature, FACE_FEATURE_VECTOR_LEN);

    ref = av_buffer_create((uint8_t *)larray, sizeof(*larray),
            &infer_labels_buffer_free, NULL, 0);

    s->labels = ref;
    av_free(dup);

    return 0;
}

static av_cold void identify_uninit(AVFilterContext *ctx)
{
    int i;
    InferenceIdentifyContext *s = ctx->priv;

    av_buffer_unref(&s->labels);

    for (i = 0; i < s->features_num; i++) {
        av_freep(&s->features[i]->feature);
        av_freep(&s->features[i]);
    }
    if (s->norm_std)
        av_free(s->norm_std);
}

static av_cold void dump_face_id(AVFilterContext *ctx, int label_id,
                                 float conf, AVBufferRef *label_buf)
{
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(ctx, AV_LOG_DEBUG,"CLASSIFY META - Face_id:%d Name:%s Conf:%1.2f\n",
           label_id, array->label[label_id], conf);
}

static int face_identify(AVFilterContext *ctx, AVFrame *frame)
{
    int i;
    InferenceIdentifyContext *s = ctx->priv;
    AVFrameSideData *side_data;
    ClassifyArray *c_array;
    InferClassificationMeta *meta;

    side_data = av_frame_get_side_data(frame,
            AV_FRAME_DATA_INFERENCE_CLASSIFICATION);

    if (!side_data)
        return 0;

    meta = (InferClassificationMeta *)side_data->data;
    if (!meta)
        return 0;

    c_array = meta->c_array;
    for (i = 0; i < c_array->num; i++) {
        int n, label = 0;
        float *vector;
        InferClassification *c;
        double dot_product, norm_feature, confidence, angle;
        double min_angle = 180.0f;

        c = c_array->classifications[i];
        vector = (float *)c->tensor_buf->data;
        norm_feature = av_norm(vector, FACE_FEATURE_VECTOR_LEN);
        for (n = 0; n < s->features_num; n++) {
            dot_product = av_dot(vector, s->features[n]->feature, FACE_FEATURE_VECTOR_LEN);

            angle = acos((dot_product - 0.0001f) / (s->norm_std[n] * norm_feature))
                    /
                    PI * 180.0;
            if (angle < 70 && angle < min_angle) {
                label = s->features[n]->label_id;
                min_angle = angle;
            }
        }

        confidence = (90.0f - min_angle) / 90.0f;

        c->label_id   = label;
        c->name       = (char *)"face_id";
        c->confidence = (float)confidence;
        c->label_buf  = av_buffer_ref(s->labels);

        dump_face_id(ctx, label, confidence, s->labels);
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx  = inlink->dst;
    AVFilterLink *outlink = inlink->dst->outputs[0];

    face_identify(ctx, in);

    return ff_filter_frame(outlink, in);
}

static const AVOption inference_identify_options[] = {
    { "gallery", "JSON file with list of image examples for each known object/face/person",
        OFFSET(gallery), AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(inference_identify);

static const AVFilterPad identify_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .filter_frame  = filter_frame,
    },
    { NULL }
};

static const AVFilterPad identify_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_inference_identify= {
    .name          = "identify",
    .description   = NULL_IF_CONFIG_SMALL("DNN Inference identification."),
    .priv_size     = sizeof(InferenceIdentifyContext),
    .query_formats = query_formats,
    .init          = identify_init,
    .uninit        = identify_uninit,
    .inputs        = identify_inputs,
    .outputs       = identify_outputs,
    .priv_class    = &inference_identify_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
