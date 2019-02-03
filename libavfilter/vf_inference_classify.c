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
 * dnn inference classify filter
 */
#include "libavutil/opt.h"
#include "libavutil/mem.h"
#include "libavutil/eval.h"
#include "libavutil/avassert.h"
#include "libavutil/pixdesc.h"
#include "libavutil/mathematics.h"

#include "formats.h"
#include "internal.h"
#include "avfilter.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"

#include "inference.h"
#include "dnn_interface.h"

#define OFFSET(x) offsetof(InferenceClassifyContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

#define PI 3.1415926
#define MAX_MODEL_NUM 8
#define FACE_FEATURE_VECTOR_LEN 256
#define THRESHOLD_RECOGNITION   70

static char string_age[]    = "age";
static char string_gender[] = "gender";

typedef int (*ClassifyInit)(AVFilterContext *ctx, size_t index);

typedef int (*ClassifyUnInit)(AVFilterContext *ctx, size_t index);

typedef int (*ClassifyProcess)(AVFilterContext*, int, int, int,
                               InferTensorMeta*, InferClassificationMeta*);

typedef struct InferenceClassifyContext {
    const AVClass *class;

    InferenceBaseContext *infer_bases[MAX_MODEL_NUM];

    char  *labels;
    char  *names;
    char  *model_file;
    char  *feature_file;    ///< binary feature file for face identification
    int    feature_num;     ///< identification face feature number
    int    loaded_num;
    int    backend_type;
    int    device_type;

    int    batch_size;
    int    every_nth_frame;

    void           *priv[MAX_MODEL_NUM];
    char           *name_array[MAX_MODEL_NUM];
    AVBufferRef    *label_bufs[MAX_MODEL_NUM];

    ClassifyInit    init[MAX_MODEL_NUM];
    ClassifyUnInit  uninit[MAX_MODEL_NUM];
    ClassifyProcess post_process[MAX_MODEL_NUM];
} InferenceClassifyContext;

typedef struct FaceIdentifyContext {
    size_t   vector_num;
    double  *norm_std;
    float  **feature_vecs;
} FaceIdentifyContext;

static void infer_labels_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    LabelsArray *labels = (LabelsArray *)data;

    for (i = 0; i < labels->num; i++)
        av_freep(&labels->label[i]);

    av_free(data);
}

static void infer_classify_metadata_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    InferClassificationMeta *meta = (InferClassificationMeta *)data;
    ClassifyArray *classes        = meta->c_array;

    if (classes) {
        for (i = 0; i < classes->num; i++) {
            InferClassification *c = classes->classifications[i];
            av_buffer_unref(&c->label_buf);
            av_freep(&c);
        }
        av_freep(&classes);
    }

    av_free(data);
}

static av_cold void dump_emotion(AVFilterContext *ctx, int label_id,
                                 float conf, AVBufferRef *label_buf)
{
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(ctx, AV_LOG_DEBUG, "CLASSIFY META - Label id:%d Emotion:%s Conf:%f\n",
           label_id, array->label[label_id], conf);
}

static int emotion_classify_result_process(AVFilterContext *ctx,
                                           int detect_id,
                                           int result_id,
                                           int model_index,
                                           InferTensorMeta *meta,
                                           InferClassificationMeta *c_meta)
{
    int i, label_id = 0;
    InferenceClassifyContext *s = ctx->priv;
    const float *emo_confidence = (float *)meta->data;
    size_t labels_num           = meta->dims[2];
    float max                   = emo_confidence[0];

    InferClassification *classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    // Get the emotion with max confidence
    for (i = 1; i < labels_num; i++)
        if (emo_confidence[i] > max) { max = emo_confidence[i]; label_id = i; }

    classify->detect_id  = detect_id;
    classify->name       = s->name_array[model_index];
    classify->label_id   = label_id;
    classify->confidence = emo_confidence[label_id];
    classify->label_buf  = av_buffer_ref(s->label_bufs[model_index]);

    dump_emotion(ctx, classify->label_id, classify->confidence, classify->label_buf);

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);

    return 0;
}

static av_cold void dump_gender(AVFilterContext *ctx, int label_id,
                                float conf, AVBufferRef *label_buf)
{
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(ctx, AV_LOG_DEBUG, "CLASSIFY META - Gender:%s Conf:%1.2f\n",
           array->label[label_id], conf);
}

static av_cold void dump_age(AVFilterContext *ctx, float age)
{
    av_log(ctx, AV_LOG_DEBUG, "CLASSIFY META - Age:%1.2f\n", age);
}

static int age_gender_classify_result_process(AVFilterContext *ctx,
                                              int detect_id,
                                              int result_id,
                                              int model_index,
                                              InferTensorMeta *meta,
                                              InferClassificationMeta *c_meta)
{
    InferenceClassifyContext *s = ctx->priv;
    const float *data = (float *)meta->data;

    InferClassification *classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    classify->detect_id  = detect_id;

    if (result_id == 0) {
        // Age
        classify->name  = string_age;
        classify->value = *data * 100.0;
        dump_age(ctx, classify->value);
    } else {
        // Gender
        classify->name       = string_gender;
        // 0 - Femal, 1 - Male
        classify->label_id   = data[0] > data[1] ? 0 : 1;
        classify->confidence = data[classify->label_id];
        classify->label_buf  = av_buffer_ref(s->label_bufs[model_index]);
        dump_gender(ctx, classify->label_id, classify->confidence, classify->label_buf);
    }

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);

    return 0;
}

static int face_identify_init(AVFilterContext *ctx, size_t index)
{
    FaceIdentifyContext *identify_ctx;
    InferenceClassifyContext *s = ctx->priv;

    int i, ret, feature_size, expected_size;
    size_t vec_size_in_bytes = sizeof(float) * FACE_FEATURE_VECTOR_LEN;

    FILE *fp = fopen(s->feature_file, "rb");
    if (!fp) {
        av_log(ctx, AV_LOG_ERROR, "Could not open feature file:%s\n", s->feature_file);
        return AVERROR(EIO);
    }

    av_assert0(index < MAX_MODEL_NUM);

    if (fseek(fp, 0, SEEK_END)) {
        av_log(ctx, AV_LOG_ERROR, "Couldn't seek to the end of feature file.\n");
        fclose(fp);
        return AVERROR(EINVAL);
    }

    feature_size = ftell(fp);

    if (feature_size == -1) {
        fclose(fp);
        av_log(ctx, AV_LOG_ERROR, "Couldn't get size of feature file.\n");
        return AVERROR(EINVAL);
    } else if (feature_size % FACE_FEATURE_VECTOR_LEN) {
        fclose(fp);
        av_log(ctx, AV_LOG_ERROR, "Feature data must align to %d.\n", FACE_FEATURE_VECTOR_LEN);
        return AVERROR(EINVAL);
    }

    if (s->feature_num > 0) {
        expected_size = s->feature_num * vec_size_in_bytes;
        if (expected_size != feature_size) {
            fclose(fp);
            av_log(ctx, AV_LOG_ERROR, "Unexpected feature file size.\n");
            return AVERROR(EINVAL);
        }
    } else {
        s->feature_num = feature_size / vec_size_in_bytes;
    }

    identify_ctx = av_mallocz(sizeof(*identify_ctx));
    if (!identify_ctx) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    identify_ctx->vector_num = s->feature_num;

    identify_ctx->feature_vecs = av_mallocz(sizeof(float *) * identify_ctx->vector_num);
    if (!identify_ctx->feature_vecs) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    rewind(fp);

    for (i = 0; i <identify_ctx->vector_num; i++) {
        identify_ctx->feature_vecs[i] = av_malloc(vec_size_in_bytes);
        if (!identify_ctx->feature_vecs[i]) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }
        if (fread(identify_ctx->feature_vecs[i], vec_size_in_bytes, 1, fp) != 1) {
            ret = AVERROR(EINVAL);
            goto fail;
        }
    }

    identify_ctx->norm_std = av_mallocz(sizeof(double) * identify_ctx->vector_num);
    if (!identify_ctx->norm_std) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    for (i = 0; i < identify_ctx->vector_num; i++)
        identify_ctx->norm_std[i] = av_norm(identify_ctx->feature_vecs[i],
                                            FACE_FEATURE_VECTOR_LEN);

    s->priv[index] = identify_ctx;
    fclose(fp);
    return 0;
fail:
    fclose(fp);

    if (identify_ctx) {
        if (identify_ctx->feature_vecs) {
            for (i = 0; i <identify_ctx->vector_num; i++) {
                if (identify_ctx->feature_vecs[i])
                    av_free(identify_ctx->feature_vecs[i]);
            }
            av_free(identify_ctx->feature_vecs);
        }
        av_free(identify_ctx);
    }
    return ret;
}

static int face_identify_uninit(AVFilterContext *ctx, size_t index)
{
    int i;
    InferenceClassifyContext *s = ctx->priv;
    FaceIdentifyContext *identify_ctx = s->priv[index];

    if (!identify_ctx) {
        av_log(ctx, AV_LOG_WARNING, "Empty face identify ctx.\n");
        return 0;
    }

    if (identify_ctx->feature_vecs) {
        for (i = 0; i < identify_ctx->vector_num; i++)
            av_free(identify_ctx->feature_vecs[i]);
        av_free(identify_ctx->feature_vecs);
    }

    if (identify_ctx->norm_std)
        av_free(identify_ctx->norm_std);

    av_free(identify_ctx);
    s->priv[index] = NULL;

    return 0;
}

static av_cold void dump_face_id(AVFilterContext *ctx, int label_id,
                                 float conf, AVBufferRef *label_buf)
{
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(ctx, AV_LOG_DEBUG,"CLASSIFY META - Face_id:%d Name:%s Conf:%1.2f\n",
           label_id, array->label[label_id], conf);
}

static int face_identify_result_process(AVFilterContext *ctx,
                                        int detect_id,
                                        int result_id,
                                        int model_index,
                                        InferTensorMeta *meta,
                                        InferClassificationMeta *c_meta)
{
    int i, label_id = 0;
    InferClassification *classify;
    double dot_product, norm_feature, confidence, *angles;
    InferenceClassifyContext *s = ctx->priv;
    FaceIdentifyContext      *f = s->priv[model_index];
    double            min_angle = 180.0f;
    float       *feature_vector = meta->data;

    angles = av_malloc(sizeof(double) * f->vector_num);
    if (!angles)
        return AVERROR(ENOMEM);

    norm_feature = av_norm(feature_vector, FACE_FEATURE_VECTOR_LEN);

    for (i = 0; i < f->vector_num; i++) {
        dot_product = av_dot(feature_vector,
                             f->feature_vecs[i],
                             FACE_FEATURE_VECTOR_LEN);

        angles[i] = acos((dot_product - 0.0001f) /
                         (f->norm_std[i] * norm_feature)) /
                    PI * 180.0;
        if (angles[i] < THRESHOLD_RECOGNITION && angles[i] < min_angle) {
            label_id  = i;
            min_angle = angles[i];
        }
    }

    confidence = (90.0f - min_angle) / 90.0f;

    av_free(angles);

    classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    classify->detect_id  = detect_id;
    classify->name       = s->name_array[model_index];
    classify->label_id   = label_id;
    classify->confidence = (float)confidence;
    classify->label_buf  = av_buffer_ref(s->label_bufs[model_index]);

    dump_face_id(ctx, label_id, confidence, s->label_bufs[model_index]);

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);

    return 0;
}

static int query_formats(AVFilterContext *context)
{
    AVFilterFormats *formats_list;
    const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_YUV422P,  AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_YUVJ422P, AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_YUV410P,  AV_PIX_FMT_YUV411P,  AV_PIX_FMT_GRAY8,
        AV_PIX_FMT_BGR24,    AV_PIX_FMT_BGRA,     AV_PIX_FMT_NONE};

    formats_list = ff_make_format_list(pixel_formats);
    if (!formats_list) {
        av_log(context, AV_LOG_ERROR, "Could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static av_cold int classify_init(AVFilterContext *ctx)
{
    InferenceClassifyContext *s = ctx->priv;
    int i, ret;
    int model_num = 0, label_num = 0, name_num = 0;
    const int max_num = MAX_MODEL_NUM;
    char  *names[MAX_MODEL_NUM] = { };
    char *models[MAX_MODEL_NUM] = { };
    char *labels[MAX_MODEL_NUM] = { };
    InferenceParam p = {};

    av_assert0(s->model_file);

    av_split(s->model_file, "&", models, &model_num, max_num);
    for (i = 0; i < model_num; i++)
        av_log(ctx, AV_LOG_INFO, "model[%d]:%s\n", i, models[i]);

    av_split(s->labels, "&", labels, &label_num, max_num);
    for (i = 0; i < label_num; i++)
        av_log(ctx, AV_LOG_INFO, "label[%d]:%s\n", i, labels[i]);

    av_split(s->names, "&", names, &name_num, max_num);
    for (i = 0; i < name_num; i++)
        av_log(ctx, AV_LOG_INFO, "name[%d]:%s\n", i, names[i]);

    av_assert0(s->backend_type == DNN_INTEL_IE);

    p.backend_type    = s->backend_type;
    p.device_type     = s->device_type;
    p.batch_size      = s->batch_size;
    p.every_nth_frame = s->every_nth_frame;
    p.input_precision = DNN_DATA_PRECISION_U8;
    p.input_layout    = DNN_DATA_LAYOUT_NCHW;
    p.input_is_image  = 1;

    for (i = 0; i < model_num; i++) {
        InferenceBaseContext *base = NULL;

        p.model_file  = models[i];
        ret = ff_inference_base_create(ctx, &base, &p);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not create inference\n");
            return ret;
        }

        s->infer_bases[i] = base;
    }
    s->loaded_num = model_num;

    for (i = 0; i < label_num; i++) {
        int n, labels_num;
        AVBufferRef *ref    = NULL;
        LabelsArray *larray = NULL;
        char buffer[4096]   = { };
        char *_labels[100]  = { };

        FILE *fp = fopen(labels[i], "rb");
        if (!fp) {
            av_log(ctx, AV_LOG_ERROR, "Could not open file:%s\n", labels[i]);
            ret = AVERROR(EIO);
            goto fail;
        }

        n = fread(buffer, sizeof(buffer), 1, fp);
        fclose(fp);

        buffer[strcspn(buffer, "\n")] = 0;
        av_split(buffer, ",", _labels, &labels_num, 100);

        larray = av_mallocz(sizeof(*larray));
        if (!larray) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        for (n = 0; n < labels_num; n++) {
            char *l = av_strdup(_labels[n]);
            av_dynarray_add(&larray->label, &larray->num, l);
        }

        ref = av_buffer_create((uint8_t *)larray, sizeof(*larray),
                               &infer_labels_buffer_free, NULL, 0);
        s->label_bufs[i] = ref;
    }

    for (i = 0; i < name_num; i++) {
        s->name_array[i] = names[i];
        if (strstr(names[i], "emotion")) {
            s->post_process[i] = &emotion_classify_result_process;
        } else if (strstr(names[i], "age") && strstr(names[i], "gend")) {
            s->post_process[i] = &age_gender_classify_result_process;
        } else if (strstr(names[i], "face")) {
            VideoPP *vpp = ff_inference_base_get_vpp(s->infer_bases[i]);

            // face reidentification model requires RGB format
            vpp->expect_format = AV_PIX_FMT_RGB24;

            s->init[i]         = &face_identify_init;
            s->uninit[i]       = &face_identify_uninit;
            s->post_process[i] = &face_identify_result_process;
        }

        if (s->init[i] && s->init[i](ctx, i) < 0)
            goto fail;
    }

    return 0;

fail:
    for (i = 0; i < model_num; i++) {
        ff_inference_base_free(&s->infer_bases[i]);
        if (s->label_bufs[i])
            av_buffer_unref(&s->label_bufs[i]);
    }

    return ret;
}

static av_cold void classify_uninit(AVFilterContext *ctx)
{
    int i;
    InferenceClassifyContext *s = ctx->priv;

    for (i = 0; i < s->loaded_num; i++) {
        if (s->uninit[i]) s->uninit[i](ctx, i);

        ff_inference_base_free(&s->infer_bases[i]);

        av_buffer_unref(&s->label_bufs[i]);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    int i, ret;
    AVFilterContext *ctx        = inlink->dst;
    InferenceClassifyContext *s = ctx->priv;
    AVFilterLink *outlink       = inlink->dst->outputs[0];
    AVBufferRef             *ref;
    AVFrameSideData         *sd, *new_sd;
    BBoxesArray             *boxes;
    InferDetectionMeta      *d_meta;
    ClassifyArray           *c_array;
    InferClassificationMeta *c_meta;

    sd = av_frame_get_side_data(in, AV_FRAME_DATA_INFERENCE_DETECTION);
    if (!sd)
        goto done;

    d_meta = (InferDetectionMeta *)sd->data;
    if (!d_meta)
        goto fail;

    boxes = d_meta->bboxes;
    if (!boxes || !boxes->num)
        goto done;

    c_meta = av_mallocz(sizeof(*c_meta));
    c_array = av_mallocz(sizeof(*c_array));
    if (!c_meta || !c_array) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    c_meta->c_array = c_array;

    // handle according to detected metadata one by one
    for (i = 0; i < boxes->num; i++) {
        int j;
        InferDetection *bbox = boxes->bbox[i];

        // process for each model
        for (j = 0; j < s->loaded_num; j++) {
            int output;
            InferenceBaseContext *base = s->infer_bases[j];

            VideoPP *vpp        = ff_inference_base_get_vpp(base);
            AVFrame *tmp        = vpp->frames[0];
            DNNModelInfo *iinfo = ff_inference_base_get_input_info(base);
            DNNModelInfo *oinfo = ff_inference_base_get_output_info(base);

            ret = vpp->crop_and_scale(in,
                                      bbox->x_min * in->width,
                                      bbox->y_min * in->height,
                                      bbox->x_max * in->width,
                                      bbox->y_max * in->height,
                                      iinfo->width[0],
                                      iinfo->height[0],
                                      vpp->expect_format,
                                      tmp->data,
                                      tmp->linesize);

            // TODO: support dynamic batch for faces
            ff_inference_base_submit_frame(base, tmp, 0, 0);
            ff_inference_base_infer(base);

            for (output = 0; output < oinfo->numbers; output++) {
                InferTensorMeta tensor_meta = { };
                ff_inference_base_get_infer_result(base, output, &tensor_meta);

                if (s->post_process[j])
                    s->post_process[j](ctx, i, output, j, &tensor_meta, c_meta);
            }
        }
    }

    ref = av_buffer_create((uint8_t *)c_meta, sizeof(*c_meta),
                           &infer_classify_metadata_buffer_free, NULL, 0);
    if (!ref)
        return AVERROR(ENOMEM);

    // add meta data to side data
    new_sd = av_frame_new_side_data_from_buf(in, AV_FRAME_DATA_INFERENCE_CLASSIFICATION, ref);
    if (!new_sd) {
        av_buffer_unref(&ref);
        av_log(NULL, AV_LOG_ERROR, "Could not add new side data\n");
        return AVERROR(ENOMEM);
    }

done:
    return ff_filter_frame(outlink, in);
fail:
    av_frame_free(&in);
    return ret;
}

static av_cold int config_input(AVFilterLink *inlink)
{
    int i, j;
    AVFilterContext      *ctx        = inlink->dst;
    InferenceClassifyContext *s      = ctx->priv;
    enum AVPixelFormat expect_format = AV_PIX_FMT_BGR24;
    const AVPixFmtDescriptor *desc   = av_pix_fmt_desc_get(inlink->format);

    for (i = 0; i < s->loaded_num; i++) {
        InferenceBaseContext *base = s->infer_bases[i];
        DNNModelInfo *info         = ff_inference_base_get_input_info(base);
        VideoPP *vpp               = ff_inference_base_get_vpp(base);

        vpp->device = (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) ?
            VPP_DEVICE_HW : VPP_DEVICE_SW;

        // allocate avframes to save preprocessed data
        for (j = 0; j < info->numbers; j++) {
            int ret;
            AVFrame *frame = av_frame_alloc();
            if (!frame)
                return AVERROR(ENOMEM);

            frame->format = expect_format;
            frame->width  = info->width[j];
            frame->height = info->height[j];

            ret = av_frame_get_buffer(frame, 0);
            if (ret < 0) {
                av_frame_free(&frame);
                return ret;
            }
            vpp->frames[j] = frame;
        }
    }

    return 0;
}

static av_cold int config_output(AVFilterLink *outlink)
{
    return 0;
}

static const AVOption inference_classify_options[] = {
    { "dnn_backend",  "DNN backend for model execution", OFFSET(backend_type),    AV_OPT_TYPE_FLAGS,  { .i64 = DNN_INTEL_IE },          0, 2,    FLAGS, "engine" },
    { "model",        "path to model files for network", OFFSET(model_file),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "label",        "labels for classify",             OFFSET(labels),          AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "name",         "classify type names",             OFFSET(names),           AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "device",       "running on device type",          OFFSET(device_type),     AV_OPT_TYPE_FLAGS,  { .i64 = DNN_TARGET_DEVICE_CPU }, 0, 12,   FLAGS },
    { "interval",     "do infer every Nth frame",        OFFSET(every_nth_frame), AV_OPT_TYPE_INT,    { .i64 = 1 },                     0, 15,   FLAGS },
    { "batch_size",   "batch size per infer",            OFFSET(batch_size),      AV_OPT_TYPE_INT,    { .i64 = 1 },                     1, 1024, FLAGS },
    { "feature_file", "registered face feature data",    OFFSET(feature_file),    AV_OPT_TYPE_STRING, { .str = NULL},                   0,    0, FLAGS, "face_identify" },
    { "feature_num",  "registered face number",          OFFSET(feature_num),     AV_OPT_TYPE_INT,    { .i64 = 0},                      0, 1024, FLAGS, "face_identify" },
    { NULL }
};

AVFILTER_DEFINE_CLASS(inference_classify);

static const AVFilterPad classify_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_input,
        .filter_frame  = filter_frame,
    },
    { NULL }
};

static const AVFilterPad classify_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_inference_classify = {
    .name          = "classify",
    .description   = NULL_IF_CONFIG_SMALL("DNN Inference classification."),
    .priv_size     = sizeof(InferenceClassifyContext),
    .query_formats = query_formats,
    .init          = classify_init,
    .uninit        = classify_uninit,
    .inputs        = classify_inputs,
    .outputs       = classify_outputs,
    .priv_class    = &inference_classify_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
