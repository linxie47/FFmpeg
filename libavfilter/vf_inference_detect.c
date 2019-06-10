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
    char  *vpp_format;
    char  *model_proc;
    int    backend_type;
    int    device_type;

    int    batch_size;
    int    frame_number;
    int    every_nth_frame;
    int    max_count;
    float  threshold;

    int    input_layout;
    int    input_precision;
    int    input_is_image;

    void  *proc_config;

    ModelInputPreproc   model_preproc;
    ModelOutputPostproc model_postproc;
} InferenceDetectContext;

static void infer_detect_metadata_buffer_free(void *opaque, uint8_t *data)
{
    BBoxesArray *bboxes = ((InferDetectionMeta *)data)->bboxes;

    if (bboxes) {
        int i;
        for (i = 0; i < bboxes->num; i++) {
            InferDetection *p = bboxes->bbox[i];
            if (p->label_buf)
                av_buffer_unref(&p->label_buf);
            av_freep(&p);
        }
        av_free(bboxes->bbox);
        av_freep(&bboxes);
    }

    av_free(data);
}

static int detect_postprocess(AVFilterContext *ctx, InferTensorMeta *meta, AVFrame *frame)
{
    int i;
    InferenceDetectContext *s = ctx->priv;
    int object_size           = meta->dims[3];
    int max_proposal_count    = meta->dims[2];
    const float *detection    = (float *)meta->data;
    AVBufferRef *ref;
    AVFrameSideData *sd;
    InferDetectionMeta *detect_meta = NULL;

    BBoxesArray *boxes        = av_mallocz(sizeof(*boxes));
    if (!boxes)
        return AVERROR(ENOMEM);

    detect_meta = av_malloc(sizeof(*detect_meta));
    if (!detect_meta) {
        av_free(boxes);
        return AVERROR(ENOMEM);
    }

    // FIXME: output object size standard??
    av_assert0(object_size == 7);

    av_assert0(meta->precision == DNN_DATA_PRECISION_FP32);

    av_assert0(meta->total_bytes >= max_proposal_count * object_size * sizeof(float));

    for (i = 0; i < max_proposal_count; i++) {
        InferDetection *new_bbox = av_mallocz(sizeof(*new_bbox));
        if (!new_bbox) {
            av_log(ctx, AV_LOG_ERROR, "Could not alloc bbox!\n");
            break;
        }

        new_bbox->label_id   = (int)detection[i * object_size + 1];
        new_bbox->confidence = detection[i * object_size + 2];
        new_bbox->x_min      = detection[i * object_size + 3];
        new_bbox->y_min      = detection[i * object_size + 4];
        new_bbox->x_max      = detection[i * object_size + 5];
        new_bbox->y_max      = detection[i * object_size + 6];

        if (new_bbox->confidence < s->threshold) {
            av_freep(&new_bbox);
            continue;
        }

        // TODO: use layer name to get proc
        if (s->model_postproc.procs[0].labels)
            new_bbox->label_buf = av_buffer_ref(s->model_postproc.procs[0].labels);

        av_dynarray_add(&boxes->bbox, &boxes->num, new_bbox);

        if (boxes->num >= s->max_count)
            break;
    }

    // dump face detected meta
    for (i = 0; i < boxes->num; i++) {
        InferDetection *p = boxes->bbox[i];
        av_log(ctx, AV_LOG_DEBUG,
               "DETECT META - label:%d confi:%f coord:%f %f %f %f\n",
               p->label_id, p->confidence,p->x_min, p->y_min, p->x_max, p->y_max);
    }

    detect_meta->bboxes = boxes;

    ref = av_buffer_create((uint8_t *)detect_meta, sizeof(*detect_meta),
                           &infer_detect_metadata_buffer_free, NULL, 0);
    if (!ref) {
        infer_detect_metadata_buffer_free(NULL, (uint8_t *)detect_meta);
        return AVERROR(ENOMEM);
    }

    // add meta data to side data
    sd = av_frame_new_side_data_from_buf(frame, AV_FRAME_DATA_INFERENCE_DETECTION, ref);
    if (!sd) {
        av_buffer_unref(&ref);
        av_log(ctx, AV_LOG_ERROR, "Could not add new side data\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}

static int detect_preprocess(InferenceBaseContext *base, int index, AVFrame *in, AVFrame **out)
{
    int ret;
    VideoPP *vpp = ff_inference_base_get_vpp(base);
    AVFrame *tmp = vpp->frames[index];

    if (vpp->device == VPP_DEVICE_SW) {
        if (!vpp->sw_vpp->scale_contexts[index]) {
            *out = in;
            return 0;
        }

        ret = vpp->sw_vpp->scale(vpp->sw_vpp->scale_contexts[index],
                (const uint8_t * const*)in->data,
                in->linesize, 0, in->height, tmp->data, tmp->linesize);
    } else {
#if CONFIG_VAAPI
        ret = vpp->va_vpp->scale(vpp->va_vpp, in,
                tmp->width, tmp->height, tmp->data, tmp->linesize);
#endif
    }
    *out = tmp;
    return ret;
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

static int config_input(AVFilterLink *inlink)
{
    int ret;
    AVFrame *frame;
    AVFilterContext      *ctx        = inlink->dst;
    InferenceDetectContext *s        = ctx->priv;
    enum AVPixelFormat expect_format = AV_PIX_FMT_BGR24;

    const AVPixFmtDescriptor *desc   = av_pix_fmt_desc_get(inlink->format);
    DNNModelInfo *info               = ff_inference_base_get_input_info(s->base);
    VideoPP *vpp                     = ff_inference_base_get_vpp(s->base);

    int width = info->dims[0][0], height = info->dims[0][1];

    ff_inference_dump_model_info(ctx, info);

    // right now, no model needs multiple inputs
    av_assert0(info->number == 1);

    if (!desc)
        return AVERROR(EINVAL);
    vpp->device = (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) ? VPP_DEVICE_HW : VPP_DEVICE_SW;

    // allocate frame to save scaled output
    frame = av_frame_alloc();
    if (!frame)
        return AVERROR(ENOMEM);
    frame->width   = width;
    frame->height  = height;
    frame->format  = expect_format;
    vpp->frames[0] = frame;

    if (vpp->device == VPP_DEVICE_SW) {
        int need_scale = expect_format != inlink->format ||
                         width         != inlink->w      ||
                         height        != inlink->h;

        if (need_scale) {
            if (av_frame_get_buffer(frame, 0) < 0) {
                av_frame_free(&frame);
                return AVERROR(ENOMEM);
            }

            vpp->sw_vpp->scale_contexts[0] = sws_getContext(
                    inlink->w, inlink->h, inlink->format,
                    width, height, expect_format,
                    SWS_BILINEAR, NULL, NULL, NULL);

            if (!vpp->sw_vpp->scale_contexts[0]) {
                av_log(ctx, AV_LOG_ERROR, "Impossible to create scale context\n");
                av_frame_free(&frame);
                return AVERROR(EINVAL);
            }
        }
    } else {
#if CONFIG_VAAPI
        vpp->va_vpp = av_mallocz(sizeof(*vpp->va_vpp));
        if (!vpp->va_vpp) {
            ret = AVERROR(ENOMEM);
            goto fail;
        }

        ret = va_vpp_device_create(vpp->va_vpp, inlink);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Create va vpp device failed\n");
            ret = AVERROR(EINVAL);
            goto fail;
        }

        ret = va_vpp_surface_alloc(vpp->va_vpp, width, height, s->vpp_format);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Create va surface failed\n");
            ret = AVERROR(EINVAL);
            goto fail;
        }

        frame->format = vpp->va_vpp->av_format;
#endif
    }

    return 0;
fail:
    av_frame_free(&frame);
#if CONFIG_VAAPI
    if (vpp->va_vpp) {
        va_vpp_device_free(vpp->va_vpp);
        av_freep(&vpp->va_vpp);
    }
#endif
    return ret;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext      *ctx = outlink->src;
    InferenceDetectContext *s = ctx->priv;
    VideoPP *vpp              = ff_inference_base_get_vpp(s->base);

    DNNModelInfo *info = ff_inference_base_get_output_info(s->base);

    ff_inference_dump_model_info(ctx, info);

#if CONFIG_VAAPI
    if (vpp->device == VPP_DEVICE_HW) {
        if (!vpp->va_vpp || !vpp->va_vpp->hw_frames_ref) {
            av_log(ctx, AV_LOG_ERROR, "The input must have a hardware frame "
                    "reference.\n");
            return AVERROR(EINVAL);
        }
        outlink->hw_frames_ctx = av_buffer_ref(vpp->va_vpp->hw_frames_ref);
        if (!outlink->hw_frames_ctx)
            return AVERROR(ENOMEM);
    }
#endif

    return 0;
}

static av_cold int detect_init(AVFilterContext *ctx)
{
    int ret;
    InferenceDetectContext *s = ctx->priv;
    InferenceParam p = {};

    av_assert0(s->model_file);

    av_assert0(s->backend_type == DNN_INTEL_IE);

    ff_load_default_model_proc(&s->model_preproc, &s->model_postproc);

    if (s->model_proc) {
        void *proc = ff_read_model_proc(s->model_proc);
        if (!proc) {
            av_log(ctx, AV_LOG_ERROR, "Could not read proc config file:"
                    "%s\n", s->model_proc);
            return AVERROR(EIO);
        }

        if (ff_parse_input_preproc(proc, &s->model_preproc) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Parse input preproc error.\n");
            return AVERROR(EIO);
        }

        if (ff_parse_output_postproc(proc, &s->model_postproc) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Parse output postproc error.\n");
            return AVERROR(EIO);
        }

        s->proc_config = proc;
    }

    p.model_file      = s->model_file;
    p.backend_type    = s->backend_type;
    p.device_type     = s->device_type;
    p.batch_size      = s->batch_size;
    p.input_precision = DNN_DATA_PRECISION_U8;
    p.input_layout    = DNN_DATA_LAYOUT_NCHW;
    p.input_is_image  = 1;
    p.preprocess      = &detect_preprocess;

    ret = ff_inference_base_create(ctx, &s->base, &p);
    if (ret < 0) {
        av_log(ctx, AV_LOG_ERROR, "Could not create inference\n");
        return ret;
    }

    return 0;
}

static av_cold void detect_uninit(AVFilterContext *ctx)
{
    InferenceDetectContext *s = ctx->priv;

    ff_inference_base_free(&s->base);

    ff_release_model_proc(s->proc_config, &s->model_preproc, &s->model_postproc);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    int ret;
    AVFilterContext *ctx      = inlink->dst;
    InferenceDetectContext *s = ctx->priv;
    AVFilterLink *outlink     = inlink->dst->outputs[0];
    InferTensorMeta tensor_meta = { };

    if (s->frame_number % s->every_nth_frame != 0)
        goto done;

    ret = ff_inference_base_filter_frame(s->base, in);
    if (ret < 0)
        goto fail;

    ret = ff_inference_base_get_infer_result(s->base, 0, &tensor_meta);
    if (ret < 0)
        goto fail;

    detect_postprocess(ctx, &tensor_meta, in);

done:
    s->frame_number++;
    return ff_filter_frame(outlink, in);
fail:
    av_frame_free(&in);
    return AVERROR(EIO);
}

static const AVOption inference_detect_options[] = {
    { "dnn_backend", "DNN backend for model execution", OFFSET(backend_type),    AV_OPT_TYPE_FLAGS,  { .i64 = DNN_INTEL_IE },          0, 2,  FLAGS, "engine" },
    { "model",       "path to model file for network",  OFFSET(model_file),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,  FLAGS },
    { "model_proc",  "model preproc and postproc",      OFFSET(model_proc),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,  FLAGS },
    { "device",      "running on device type",          OFFSET(device_type),     AV_OPT_TYPE_FLAGS,  { .i64 = DNN_TARGET_DEVICE_CPU }, 0, 12, FLAGS },
    { "vpp_format",  "specify vpp output format",       OFFSET(vpp_format),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,  FLAGS },
    { "interval",    "detect every Nth frame",          OFFSET(every_nth_frame), AV_OPT_TYPE_INT,    { .i64 = 1 },   1, 1024,    FLAGS},
    { "batch_size",  "batch size per infer",            OFFSET(batch_size),      AV_OPT_TYPE_INT,    { .i64 = 1 },   1, 1024,    FLAGS},
    { "max_count",   "max count of output result",      OFFSET(max_count),       AV_OPT_TYPE_INT,    { .i64 = 1000}, 1, INT_MAX, FLAGS},
    { "threshold",   "threshod to filter output data",  OFFSET(threshold),       AV_OPT_TYPE_FLOAT,  { .dbl = 0.5},  0, 1,       FLAGS},

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
