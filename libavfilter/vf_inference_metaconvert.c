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
#include "libavutil/avassert.h"

#include "formats.h"
#include "internal.h"
#include "avfilter.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"

#include "inference_backend/ff_base_inference.h"
#include "inference_backend/metaconverter.h"

static int query_formats(AVFilterContext *ctx) {
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

static av_cold int metaconvert_init(AVFilterContext *ctx) {
    MetaConvertContext *s = ctx->priv;
    char filename[1024] = {0};

    if (!s->converter || !s->method || !s->location) {
        av_log(ctx, AV_LOG_ERROR, "Missing key parameters!!\n");
        return AVERROR(EINVAL);
    }

    snprintf(filename, sizeof(filename), "%s/%s.json", s->location, s->method);
    s->f = fopen(filename, "wb");
    if (!s->f) {
        av_log(ctx, AV_LOG_ERROR, "Failed to open/create file: %s\n", filename);
        return AVERROR(EINVAL);
    }

    av_log(ctx, AV_LOG_INFO, "\nconverter:%s\nmethod:%s\nlocation:%s\n", s->converter, s->method, s->location);

    if (!strcmp(s->converter, "classification-to-json")) {
        s->convert_func = &classification_to_json;
    } else if (!strcmp(s->converter, "detection-to-json")) {
        s->convert_func = &detection_to_json;
    } else if (!strcmp(s->converter, "all-to-json")) {
        s->convert_func = &all_to_json;
    } else if (!strcmp(s->converter, "tensors-to-file")) {
        s->convert_func = &tensors_to_file;
    } else {
        av_log(ctx, AV_LOG_ERROR, "Please check your converter!");
        return AVERROR(ERANGE);
    }

    return 0;
}

static av_cold void metaconvert_uninit(AVFilterContext *ctx) {
    MetaConvertContext *s = ctx->priv;
    const char *tail = "\n}";

    fwrite(tail, sizeof(char), strlen(tail) / sizeof(char), s->f);
    fclose(s->f);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in) {
    AVFilterContext *ctx  = inlink->dst;
    MetaConvertContext *s = ctx->priv;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    const char *proc_json = NULL;
    const char *str_insert = ",\n\"Metadata\":";
    const char *head = "{\n\"Metadata\":";
    json_object *info_object = json_object_new_object();
    int ret;

    ret = s->convert_func(ctx, in, info_object);
    if (ret) {
        int64_t nano_ts = 1000000000;
        char timestamp[1024] = {0};

        nano_ts = in->pts * (nano_ts * inlink->time_base.num / inlink->time_base.den);
        snprintf(timestamp, sizeof(timestamp), "%"PRIu64"", nano_ts);

        json_object_object_add(info_object, "timestamp", json_object_new_string(timestamp));

        if(s->frame_number == 0) {
            fwrite(head, sizeof(char), strlen(head) / sizeof(char), s->f);
        } else {
            fwrite(str_insert, sizeof(char), strlen(str_insert) / sizeof(char), s->f);
        }

        proc_json = json_object_to_json_string_ext(info_object, JSON_C_TO_STRING_PRETTY);
        fwrite(proc_json, sizeof(char), strlen(proc_json) / sizeof(char), s->f);

        s->frame_number++;
    }
    json_object_put(info_object);

    return ff_filter_frame(outlink, in);
}

static const AVOption inference_metaconvert_options[] = {
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
    .uninit        = metaconvert_uninit,
    .inputs        = metaconvert_inputs,
    .outputs       = metaconvert_outputs,
    .priv_class    = &inference_metaconvert_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE,
};
