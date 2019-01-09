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
 * dnn inference detection filter
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

#define OFFSET(x) offsetof(InferenceDetectContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

#define FUNC_ENTRY() printf("enter >>> %s\n", __FUNCTION__);
#define FUNC_EXIT()  printf("exit  <<< %s\n", __FUNCTION__);

typedef struct InferenceDetectContext {
    const AVClass *class;

    InferenceBaseContext *base;

    char  *model_file;
    int    backend_type;
    int    device_type;

    int    batch_size;
    int    every_nth_frame;
    float  threshold;

    int    input_layout;
    int    input_precision;
    int    input_is_image;

    char  *name;
    char  *params;
    int  (*init)  (AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    int  (*end_frame_filter)(AVFilterContext *ctx, InferTensorMeta *data, AVFrame *frame);
    void  *priv;
} InferenceDetectContext;

static void infer_detect_metadata_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    InferDetectionMeta *meta = (InferDetectionMeta *)data;
    LabelsArray *labels = meta->labels;
    BBoxesArray *bboxes = meta->bboxes;

    if (bboxes) {
        for (i = 0; i < bboxes->num; i++) {
            InferDetection *p = bboxes->bbox[i];
            av_freep(&p);
        }
        av_freep(&bboxes);
    }

    if (labels) {
        for (i = 0; i < labels->num; i++) {
            char *l = labels->label[i];
            av_freep(&l);
        }
        av_freep(&labels);
    }

    av_free(data);
}

typedef struct FaceDetectContext {
    int max_num;

} FaceDetectContext;

static int  face_init(AVFilterContext *ctx, const char *args) {return 0;}
static void face_uninit(AVFilterContext *ctx) {}

static int  face_end_frame_filter(AVFilterContext *ctx, InferTensorMeta *meta, AVFrame *frame)
{
    int i;
    InferenceDetectContext *s = ctx->priv;
    int object_size           = meta->dims[0];
    int max_proposal_count    = meta->dims[1];
    const float *detection    = (float *)meta->data;
    AVBufferRef *ref;
    AVFrameSideData *sd;
    InferDetectionMeta *detect_meta = NULL;

    BBoxesArray *boxes        = av_mallocz(sizeof(BBoxesArray));
    if (!boxes)
        return AVERROR(ENOMEM);

    detect_meta = av_malloc(sizeof(InferDetectionMeta));
    if (!detect_meta)
        return AVERROR(ENOMEM);

    // FIXME: output object size standard??
    av_assert0(object_size == 7);

    av_assert0(meta->precision == DNN_DATA_PRECISION_FP32);

    av_assert0(meta->total_bytes >= max_proposal_count * object_size * sizeof(float));

    for (i = 0; i < max_proposal_count; i++) {
        InferDetection *new_bbox = av_mallocz(sizeof(InferDetection));

        new_bbox->label_id   = (int)detection[i * object_size + 1];
        new_bbox->confidence = detection[i * object_size + 2];
        new_bbox->x_min      = detection[i * object_size + 3];
        new_bbox->y_min      = detection[i * object_size + 4];
        new_bbox->x_max      = detection[i * object_size + 5];
        new_bbox->y_max      = detection[i * object_size + 6];

        if (new_bbox->confidence < s->threshold) {
            av_freep(&new_bbox);
            break;
        }

        av_dynarray_add(&boxes->bbox, &boxes->num, new_bbox);
    }

    // dump face detected meta
    for (i = 0; i < boxes->num; i++) {
        InferDetection *p = boxes->bbox[i];
        av_log(ctx, AV_LOG_DEBUG, "DETECT META - label:%d confi:%f coord:%f %f %f %f\n",
               p->label_id, p->confidence, p->x_min, p->y_min, p->x_max, p->y_max);
    }

    ref = av_buffer_create((uint8_t *)detect_meta, sizeof(*detect_meta),
                           &infer_detect_metadata_buffer_free, NULL, 0);
    if (!ref)
        return AVERROR(ENOMEM);

    detect_meta->bboxes = boxes;
    detect_meta->labels = NULL;

    // add meta data to side data
    sd = av_frame_new_side_data_from_buf(frame, AV_FRAME_DATA_INFERENCE_DETECTION, ref);
    if (!sd) {
        av_buffer_unref(&ref);
        av_log(NULL, AV_LOG_ERROR, "could not add new side data\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}

typedef struct EmotionDetectContext {
    int max_num;

} EmotionDetectContext;
static int  emotion_init(AVFilterContext *ctx, const char *args) {return 0;}
static void emotion_uninit(AVFilterContext *ctx) {}
static int  emotion_end_frame_filter(AVFilterContext *ctx, InferTensorMeta *data, AVFrame *frame) { return 0; }

typedef struct LogoDetectContext {
    int max_num;

} LogoDetectContext;
static int  logo_init(AVFilterContext *ctx, const char *args) {return 0;}
static void logo_uninit(AVFilterContext *ctx) {}
static int  logo_end_frame_filter(AVFilterContext *ctx, InferTensorMeta *data, AVFrame *frame) { return 0; }

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
        av_log(context, AV_LOG_ERROR, "could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static int config_input(AVFilterLink *inlink)
{
    int i;
    AVFilterContext      *ctx        = inlink->dst;
    InferenceDetectContext *s        = ctx->priv;
    enum AVPixelFormat expect_format = AV_PIX_FMT_BGR24;

    const AVPixFmtDescriptor *desc   = av_pix_fmt_desc_get(inlink->format);
    DNNModelInfo *info               = ff_inference_base_get_input_info(s->base);
    VideoPP *vpp                     = ff_inference_base_get_vpp(s->base);

    for (i = 0; i < info->numbers; i++) {
        av_log(ctx, AV_LOG_DEBUG, "input info [%d] %d - %d %d %d - %d %d %d\n",
               i, info->batch_size, info->width[i], info->height[i], info->channels[i],
               info->is_image[i], info->precision[i], info->layout[i]);
    }

    vpp->device = (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) ? VPP_DEVICE_HW : VPP_DEVICE_SW;

    // TODO: now just handle sw vpp
    for (i = 0; i < info->numbers; i++) {
        if (expect_format   != inlink->format ||
            info->width[i]  != inlink->w      ||
            info->height[i] != inlink->h)
        {
            int ret;
            AVFrame *frame;

            vpp->scale_contexts[i] = sws_getContext(
                inlink->w,      inlink->h,       inlink->format,
                info->width[i], info->height[i], expect_format,
                SWS_BILINEAR, NULL, NULL, NULL);

            if (!vpp->scale_contexts[i]) {
                av_log(ctx, AV_LOG_ERROR, "Impossible to create scale context");
                return AVERROR(EINVAL);
            }

            frame = av_frame_alloc();
            if (!frame)
                return AVERROR(ENOMEM);

            frame->format = expect_format;
            frame->width  = info->width[i];
            frame->height = info->height[i];

            ret = av_frame_get_buffer(frame, 0);
            if (ret < 0) {
                av_frame_free(&frame);
                return ret;
            }
            vpp->frames[i] = frame;
        }
    }

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext      *ctx = outlink->src;
    InferenceDetectContext *s = ctx->priv;

    DNNModelInfo *info = ff_inference_base_get_output_info(s->base);

    for (int i = 0; i < info->numbers; i++) {
        av_log(ctx, AV_LOG_DEBUG, "output info [%d] %d - %d %d %d - %d %d %d\n",
            i, info->batch_size,
            info->width[i], info->height[i], info->channels[i],
            info->is_image[i], info->precision[i], info->layout[i]);
    }

    // TODO: define how to handle model output data

    return 0;
}

typedef struct DetectFilterEntry {
    const char *name;
    size_t priv_size;
    int  (*init)(AVFilterContext *ctx, const char *args);
    void (*uninit)(AVFilterContext *ctx);
    int  (*end_frame_filter)(AVFilterContext *ctx, InferTensorMeta *data, AVFrame *frame);
} DetectFilterEntry;

static const DetectFilterEntry detect_filter_entries[] = {
    { "face",    sizeof(FaceDetectContext),    face_init,    face_uninit,    face_end_frame_filter },
    { "emotion", sizeof(EmotionDetectContext), emotion_init, emotion_uninit, emotion_end_frame_filter  },
    { "logo",    sizeof(LogoDetectContext),    logo_init,    logo_uninit,    logo_end_frame_filter },
};

static av_cold int detect_init(AVFilterContext *ctx)
{
    int i, ret;
    InferenceDetectContext *s = ctx->priv;
    InferenceParam p = {};

    av_assert0(s->model_file && s->name);

    for (i = 0; i < FF_ARRAY_ELEMS(detect_filter_entries); i++) {
        const DetectFilterEntry *entry = &detect_filter_entries[i];
        if (!strcmp(s->name, entry->name)) {
            s->init             = entry->init;
            s->uninit           = entry->uninit;
            s->end_frame_filter = entry->end_frame_filter;

            if (!(s->priv = av_mallocz(entry->priv_size)))
                return AVERROR(ENOMEM);
        }
    }

    av_assert0(s->init);

    av_assert0(s->backend_type == DNN_INTEL_IE);

    p.model_file      = s->model_file;
    p.backend_type    = s->backend_type;
    p.device_type     = s->device_type;
    p.batch_size      = s->batch_size;
    p.every_nth_frame = s->every_nth_frame;
    p.threshold       = s->threshold;
    p.input_precision = DNN_DATA_PRECISION_U8;
    p.input_layout    = DNN_DATA_LAYOUT_NCHW;
    p.input_is_image  = 1;

    ret = ff_inference_base_create(ctx, &s->base, &p);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "could not create inference\n");
        return ret;
    }

    ret = s->init(ctx, s->params);
    if (ret < 0) {
        ff_inference_base_free(&s->base);
        av_log(ctx, AV_LOG_ERROR, "init '%s' failed\n", s->name);
        return ret;
    }

    return 0;
}

static av_cold void detect_uninit(AVFilterContext *ctx)
{
    InferenceDetectContext *s = ctx->priv;

    ff_inference_base_free(&s->base);

    av_freep(&s->priv);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    int ret;
    AVFilterContext *ctx      = inlink->dst;
    InferenceDetectContext *s = ctx->priv;
    AVFilterLink *outlink     = inlink->dst->outputs[0];
    InferTensorMeta tensor_meta = { };

    ret = ff_inference_base_filter_frame(s->base, in);
    if (ret < 0)
        goto fail;

    ret = ff_inference_base_get_infer_result(s->base, &tensor_meta);
    if (ret < 0)
        goto fail;

    s->end_frame_filter(ctx, &tensor_meta, in);

    return ff_filter_frame(outlink, in);
fail:
    av_frame_free(&in);
    return AVERROR(EIO);
}

static const AVOption inference_detect_options[] = {
    { "dnn_backend", "DNN backend for model execution", OFFSET(backend_type),    AV_OPT_TYPE_FLAGS,  { .i64 = DNN_INTEL_IE },          0, 2,  FLAGS, "engine" },
    { "model",       "path to model file for network",  OFFSET(model_file),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,  FLAGS },
    { "device",      "running on device type",          OFFSET(device_type),     AV_OPT_TYPE_FLAGS,  { .i64 = DNN_TARGET_DEVICE_CPU }, 0, 12, FLAGS },
    { "interval",    "detect every Nth frame",          OFFSET(every_nth_frame), AV_OPT_TYPE_INT,    { .i64 = 1 }, 0, 15, FLAGS},
    { "batch_size",  "batch size per infer",            OFFSET(batch_size),      AV_OPT_TYPE_INT,    { .i64 = 1 }, 0, 1024, FLAGS},
    { "threshold",   "threshod to filter output data",  OFFSET(threshold),       AV_OPT_TYPE_FLOAT,  { .dbl = 0.5}, 0, 1, FLAGS},

    { "name",        "detection type name",             OFFSET(name),            AV_OPT_TYPE_STRING, .flags = FLAGS, "detection" },
    { "filter_params", NULL,                            OFFSET(params),          AV_OPT_TYPE_STRING, .flags = FLAGS, "detection" },
    { NULL }
};

AVFILTER_DEFINE_CLASS(inference_detect);

static const AVFilterPad detect_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_input,
        .filter_frame  = filter_frame,
    },
    { NULL }
};

static const AVFilterPad detect_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_inference_detect = {
    .name          = "detect",
    .description   = NULL_IF_CONFIG_SMALL("DNN Inference detection."),
    .priv_size     = sizeof(InferenceDetectContext),
    .query_formats = query_formats,
    .init          = detect_init,
    .uninit        = detect_uninit,
    .inputs        = detect_inputs,
    .outputs       = detect_outputs,
    .priv_class    = &inference_detect_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
