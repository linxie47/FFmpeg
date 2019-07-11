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
 * image inference filter used for object classification
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
#include "filters.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/time.h"

#include "inference_backend/ff_base_inference.h"

#define OFFSET(x) offsetof(IEClassifyContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

static int flush_frame(AVFilterContext *ctx, AVFilterLink *outlink, int64_t pts, int64_t *out_pts);

typedef struct IEClassifyContext {
    const AVClass *class;

    FFBaseInference *base;

    FF_INFERENCE_OPTIONS

    int    async_preproc;
    int    backend_type;
    int    already_flushed;
} IEClassifyContext;

static int query_formats(AVFilterContext *context)
{
    AVFilterFormats *formats_list;
    const enum AVPixelFormat pixel_formats[] = {
        AV_PIX_FMT_YUV420P,  AV_PIX_FMT_NV12,
        AV_PIX_FMT_BGR24,    AV_PIX_FMT_BGRA,
        AV_PIX_FMT_BGR0,     AV_PIX_FMT_RGBP,
        AV_PIX_FMT_BGRA,     AV_PIX_FMT_VAAPI,
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
    int ret = 0;
    AVFilterContext *ctx = inlink->dst;
    IEClassifyContext *s = ctx->priv;
    const AVPixFmtDescriptor *desc   = av_pix_fmt_desc_get(inlink->format);
    if (desc == NULL)
        return AVERROR(EINVAL);

    FFInferenceParam param = { };

    av_assert0(s->model);

    param.model           = s->model;
    param.device          = s->device;
    param.nireq           = s->nireq;
    param.batch_size      = s->batch_size;
    param.every_nth_frame = s->every_nth_frame;
    param.threshold       = s->threshold;
    param.is_full_frame   = 0;
    param.infer_config    = s->infer_config;
    param.model_proc      = s->model_proc;
    param.opaque          = s->async_preproc ? (void *)MOCKER_PRE_PROC_MAGIC : 0;

    s->base = av_base_inference_create(ctx->filter->name);
    if (!s->base) {
        av_log(ctx, AV_LOG_ERROR, "Could not create inference.\n");
        return AVERROR(EINVAL);
    }

    if (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) {
        AVHWFramesContext *hw_frm_ctx = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
        AVHWDeviceContext *dev_ctx = (AVHWDeviceContext *)hw_frm_ctx->device_ref->data;
#if CONFIG_VAAPI
        param.vpp_device = VPP_DEVICE_HW;
        param.opaque = (void *)((AVVAAPIDeviceContext *)dev_ctx->hwctx)->display;
#endif
        for (int i = 0; i < ctx->nb_outputs; i++) {
            if (!ctx->outputs[i]->hw_frames_ctx)
                ctx->outputs[i]->hw_frames_ctx = av_buffer_ref(inlink->hw_frames_ctx);
        }
    }

    ret = av_base_inference_set_params(s->base, &param);

    return ret;
}

static av_cold int classify_init(AVFilterContext *ctx)
{
    /* moved to config_input */
    return 0;
}

static av_cold void classify_uninit(AVFilterContext *ctx)
{
    IEClassifyContext *s = ctx->priv;

    flush_frame(ctx, NULL, 0LL, NULL);

    av_base_inference_release(s->base);
}

static int flush_frame(AVFilterContext *ctx, AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    int ret = 0;
    IEClassifyContext *s = ctx->priv;

    if (s->already_flushed)
        return ret;

    while (!av_base_inference_frame_queue_empty(ctx, s->base)) {
        AVFrame *output = NULL;
        av_base_inference_get_frame(ctx, s->base, &output);
        if (output) {
            if (outlink) {
                ret = ff_filter_frame(outlink, output);
                if (out_pts)
                    *out_pts = output->pts + pts;
            } else {
                av_frame_free(&output);
            }
        }

        av_base_inference_send_event(ctx, s->base, INFERENCE_EVENT_EOS);
        av_usleep(5000);
    }

    s->already_flushed = 1;
    return ret;
}

static int classify_num_of_roi(AVFrame *frame) {
    int roi_num = 0;
    BBoxesArray *bboxes = NULL;
    InferDetectionMeta *detect_meta = NULL;
    AVFrameSideData *side_data;

    if (!frame)
        return 0;

    side_data = av_frame_get_side_data(frame, AV_FRAME_DATA_INFERENCE_DETECTION);
    if (side_data) {
        detect_meta = (InferDetectionMeta *)(side_data->data);
        av_assert0(detect_meta);
        bboxes = detect_meta->bboxes;
        roi_num = bboxes ? bboxes->num : 0;
    }

    return roi_num;
}

static int load_balance(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    IEClassifyContext *s = ctx->priv;
    AVFrame *in = NULL, *output = NULL;
    int64_t pts;
    int i, ret, status, idx;
    int needed, resource, prepared, got_frames = 0;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    while (av_base_inference_get_frame(ctx, s->base, &output) == 0) {
        if (output) {
            status = ff_filter_frame(outlink, output);
            if (status < 0)
                return status;
            got_frames = 1;
            output = NULL;
        }
    }

    status = ff_outlink_get_status(inlink);
    if (status)
        needed = ff_inlink_queued_frames(inlink);
    else {
        needed = 0;
        prepared = 0;
        idx = ff_inlink_queued_frames(inlink);
        resource = av_base_inference_resource_status(ctx, s->base);
        for (i = 0; i < idx; i++) {
            int num_roi;
            in = ff_inlink_peek_frame(inlink, i);
            num_roi = classify_num_of_roi(in);
            if (num_roi > s->nireq * s->batch_size) {
                needed = resource < s->nireq ? 0 : 1;
                break;
            }
            else {
                if (num_roi == 0 || (resource - prepared) >= num_roi) {
                    needed++;
                    prepared += num_roi;
                    continue;
                } else
                    break;
            }
        }
    }

    while (needed > 0) {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0) {
            av_base_inference_send_frame(ctx, s->base, in);
        }
        needed--;
    }

    if (!status && got_frames)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;

            av_log(ctx, AV_LOG_INFO, "Get EOS.\n");
            ret = flush_frame(ctx, outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    IEClassifyContext *s = ctx->priv;
    AVFrame *in = NULL, *output = NULL;
    int64_t pts;
    int ret, status;
    int got_frames = 0;

    if (av_load_balance_get())
        return load_balance(ctx);

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    // drain all frames from inlink
    do {
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0)
            av_base_inference_send_frame(ctx, s->base, in);

        while (av_base_inference_get_frame(ctx, s->base, &output) == 0) {
            if (output) {
                status = ff_filter_frame(outlink, output);
                if (status < 0)
                    return status;
                got_frames = 1;
                output = NULL;
            }
        }
    } while (ret > 0);

    if (got_frames)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;

            av_log(ctx, AV_LOG_INFO, "Get EOS.\n");
            ret = flush_frame(ctx, outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static const AVOption inference_classify_options[] = {
    { "dnn_backend",  "DNN backend for model execution", OFFSET(backend_type),    AV_OPT_TYPE_FLAGS,  { .i64 = 1},          0, 2,  FLAGS },
    { "model",        "path to model file for network",  OFFSET(model),           AV_OPT_TYPE_STRING, { .str = NULL},       0, 0,  FLAGS },
    { "model_proc",   "model preproc and postproc",      OFFSET(model_proc),      AV_OPT_TYPE_STRING, { .str = NULL},       0, 0,  FLAGS },
    { "object_class", "objective class",                 OFFSET(object_class),    AV_OPT_TYPE_STRING, { .str = NULL},       0, 0,  FLAGS },
    { "device",       "running on device name",          OFFSET(device),          AV_OPT_TYPE_STRING, { .str = NULL},       0, 0,  FLAGS },
    { "configs",      "configurations to backend",       OFFSET(infer_config),    AV_OPT_TYPE_STRING, { .str = NULL},       0, 0,  FLAGS },
    { "interval",     "detect every Nth frame",          OFFSET(every_nth_frame), AV_OPT_TYPE_INT,    { .i64 = 1 },  1, 1024, FLAGS},
    { "nireq",        "inference request number",        OFFSET(nireq),           AV_OPT_TYPE_INT,    { .i64 = 1 },  1, 128,  FLAGS},
    { "batch_size",   "batch size per infer",            OFFSET(batch_size),      AV_OPT_TYPE_INT,    { .i64 = 1 },  1, 1000, FLAGS},
    { "threshold",    "threshod to filter output data",  OFFSET(threshold),       AV_OPT_TYPE_FLOAT,  { .dbl = 0.5}, 0, 1,    FLAGS},
    { "async_preproc", "do asynchronous preproc in inference backend", OFFSET(async_preproc), AV_OPT_TYPE_BOOL, { .i64 = 0 }, 0, 1, FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(inference_classify);

static const AVFilterPad classify_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_input,
    },
    { NULL }
};

static const AVFilterPad classify_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_inference_classify = {
    .name          = "classify",
    .description   = NULL_IF_CONFIG_SMALL("Image Inference classify filter."),
    .priv_size     = sizeof(IEClassifyContext),
    .query_formats = query_formats,
    .activate      = activate,
    .init          = classify_init,
    .uninit        = classify_uninit,
    .inputs        = classify_inputs,
    .outputs       = classify_outputs,
    .priv_class    = &inference_classify_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
