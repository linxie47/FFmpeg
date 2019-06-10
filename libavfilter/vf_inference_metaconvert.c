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
 * dnn inference metadata convert filter
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

#include "inference.h"
#include "dnn_interface.h"

#define OFFSET(x) offsetof(MetaConvertContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

typedef struct MetaConvertContext {
    const AVClass *class;

    char *model;
    char *converter;
    char *method;
    char *location;
    char *layer;

    void (*convert_func)(AVFilterContext *ctx, AVFrame *frame);

} MetaConvertContext;

static int query_formats(AVFilterContext *ctx)
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
        av_log(ctx, AV_LOG_ERROR, "Could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(ctx, formats_list);
}

static av_cold void tensors_to_file(AVFilterContext *ctx, AVFrame *frame)
{
    AVFrameSideData *sd;
    MetaConvertContext *s = ctx->priv;
    InferClassificationMeta *c_meta;

    static uint32_t frame_num = 0;

    if (!(sd = av_frame_get_side_data(frame, AV_FRAME_DATA_INFERENCE_CLASSIFICATION)))
        return;

    c_meta = (InferClassificationMeta *)sd->data;

    if (c_meta) {
        int i;
        uint32_t index = 0;
        char filename[1024] = {0};
        const int meta_num = c_meta->c_array->num;
        for (i = 0; i < meta_num; i++) {
            FILE *f = NULL;
            InferClassification *c = c_meta->c_array->classifications[i];
            //TODO:check model and layer
            if (!c->tensor_buf || !c->tensor_buf->data)
                continue;

            snprintf(filename, sizeof(filename), "%s/%s_frame_%u_idx_%u.tensor", s->location,
                    s->method, frame_num, index);
            f = fopen(filename, "wb");
            if (!f) {
                av_log(ctx, AV_LOG_WARNING, "Failed to open/create file: %s\n", filename);
            } else {
                fwrite(c->tensor_buf->data, sizeof(float), c->tensor_buf->size / sizeof(float), f);
                fclose(f);
            }
            index++;
        }
    }

    frame_num++;
}

static av_cold int metaconvert_init(AVFilterContext *ctx)
{
    MetaConvertContext *s = ctx->priv;

    if (!s->model || !s->converter || !s->method) {
        av_log(ctx, AV_LOG_ERROR, "Missing key parameters!!\n");
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_INFO, "\nmodel:%s\nconverter:%s\nmethod:%s\nlocation:%s\n",
           s->model, s->converter, s->method, s->location);

    if (!strcmp(s->converter, "tensors-to-file")) {
        if (!s->location) {
            av_log(ctx, AV_LOG_ERROR, "Missing parameters location!");
            return AVERROR(EINVAL);
        }
        s->convert_func = &tensors_to_file;
    }

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx  = inlink->dst;
    MetaConvertContext *s = ctx->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];

    if (s->convert_func)
        s->convert_func(ctx, in);

    return ff_filter_frame(outlink, in);
}

static const AVOption inference_metaconvert_options[] = {
    { "model",     "select tensor by model name", OFFSET(model),     AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },
    { "layer",     "select tensor by layer name", OFFSET(layer),     AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },
    { "converter", "metadata conversion group",   OFFSET(converter), AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },
    { "method",    "metadata conversion method",  OFFSET(method),    AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },
    { "location",  "location for output files",   OFFSET(location),  AV_OPT_TYPE_STRING, { .str = NULL}, 0, 0, FLAGS },

    { NULL }
};

AVFILTER_DEFINE_CLASS(inference_metaconvert);

static const AVFilterPad metaconvert_inputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .filter_frame  = filter_frame,
    },
    { NULL }
};

static const AVFilterPad metaconvert_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_inference_metaconvert = {
    .name          = "metaconvert",
    .description   = NULL_IF_CONFIG_SMALL("DNN Inference metaconvert."),
    .priv_size     = sizeof(MetaConvertContext),
    .query_formats = query_formats,
    .init          = metaconvert_init,
    .inputs        = metaconvert_inputs,
    .outputs       = metaconvert_outputs,
    .priv_class    = &inference_metaconvert_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
