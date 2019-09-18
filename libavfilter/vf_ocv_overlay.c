/*
 * Copyright (c) 2018 Pengfei Qu
 * Copyright (c) 2019 Lin Xie
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

/**
 * @file
 * libopencv wrapper functions to overlay
 */

#include "config.h"
#if HAVE_OPENCV2_CORE_CORE_C_H
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#else
#include <opencv/cv.h>
#include <opencv/cxcore.h>
#endif
#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/file.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "inference_backend/ff_base_inference.h"

static void fill_iplimage_from_frame(IplImage *img, const AVFrame *frame, enum AVPixelFormat pixfmt)
{
    IplImage *tmpimg;
    int depth, channels_nb;

    if      (pixfmt == AV_PIX_FMT_GRAY8) { depth = IPL_DEPTH_8U;  channels_nb = 1; }
    else if (pixfmt == AV_PIX_FMT_BGRA)  { depth = IPL_DEPTH_8U;  channels_nb = 4; }
    else if (pixfmt == AV_PIX_FMT_BGR24) { depth = IPL_DEPTH_8U;  channels_nb = 3; }
    else return;

    tmpimg = cvCreateImageHeader((CvSize){frame->width, frame->height}, depth, channels_nb);
    *img = *tmpimg;
    img->imageData = img->imageDataOrigin = frame->data[0];
    img->dataOrder = IPL_DATA_ORDER_PIXEL;
    img->origin    = IPL_ORIGIN_TL;
    img->widthStep = frame->linesize[0];
}

static void fill_frame_from_iplimage(AVFrame *frame, const IplImage *img, enum AVPixelFormat pixfmt)
{
    frame->linesize[0] = img->widthStep;
    frame->data[0]     = img->imageData;
}

static int query_formats(AVFilterContext *ctx)
{
    const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_BGR24, AV_PIX_FMT_BGRA, AV_PIX_FMT_GRAY8, AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    if (!fmts_list)
        return AVERROR(ENOMEM);
    return ff_set_common_formats(ctx, fmts_list);
}

typedef struct OCVOverlayContext {
    const AVClass *class;

    char *color_line_str;
    char *color_text_str;
    int x, y, width, height;

    CvPoint lt,rd;
    CvScalar color_line;
    CvScalar color_text;
    CvFont font;
    int thickness;
    int line_type;
    int shift;
} OCVOverlayContext;

static av_cold int init(AVFilterContext *ctx)
{
    OCVOverlayContext *s = ctx->priv;

    if (strcmp(s->color_line_str, "red") == 0)
        s->color_line = cvScalar(255, 0, 0, 255);
    else if (strcmp(s->color_line_str, "green") == 0)
        s->color_line = cvScalar(0, 255, 0, 255);
    else if (strcmp(s->color_line_str, "blue") == 0)
        s->color_line = cvScalar(0, 0, 255, 255);
    else
        s->color_line = cvScalar(255, 0, 0, 255);

    if (strcmp(s->color_text_str, "red") == 0)
        s->color_text = cvScalar(255, 0, 0, 255);
    else if (strcmp(s->color_text_str, "green") == 0)
        s->color_text = cvScalar(0, 255, 0, 255);
    else if (strcmp(s->color_text_str, "blue") == 0)
        s->color_text = cvScalar(0, 0, 255, 255);
    else
        s->color_text = cvScalar(0, 0, 255, 255);

    s->line_type = 8;
    s->shift     = 0;
    cvInitFont(&s->font, CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, s->line_type);

    return 0;
}

static void rectangle(AVFilterContext *ctx, IplImage *img, CvPoint pt1, CvPoint pt2)
{
    OCVOverlayContext *s = ctx->priv;
    cvRectangle(img, pt1, pt2, s->color_line, s->thickness, s->line_type, s->shift);
}

static void put_text(AVFilterContext *ctx, IplImage *img, CvPoint pt1, const char * text)
{
    OCVOverlayContext *s = ctx->priv;
    cvPutText(img, text, pt1, &s->font, s->color_text);
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx  = inlink->dst;
    AVFilterLink *outlink = inlink->dst->outputs[0];
    AVFrameSideData *sd;
    IplImage img;
    CvPoint pt1, pt2;

    if (in->nb_side_data == 0)
        return ff_filter_frame(outlink, in);

    fill_iplimage_from_frame(&img, in , inlink->format);

    sd = av_frame_get_side_data(in, AV_FRAME_DATA_INFERENCE_DETECTION);
    if (sd) {
        InferDetectionMeta *meta = (InferDetectionMeta *)sd->data;

        BBoxesArray *boxes = meta->bboxes;
        if (boxes && boxes->num > 0) {
            int i;
            for (i = 0; i < boxes->num; i++) {
                InferDetection *box = boxes->bbox[i];

                pt1.x = box->x_min;
                pt1.y = box->y_min;
                pt2.x = box->x_max;
                pt2.y = box->y_max;

                rectangle(ctx, &img, pt1, pt2);

                // TODO: label
            }
        }
    }

    fill_frame_from_iplimage(in, &img, inlink->format);

    return ff_filter_frame(outlink, in);
}

#define OFFSET(x) offsetof(OCVOverlayContext, x)
#define FLAGS AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM
static const AVOption ocv_overlay_options[] = {
    { "color_line",     "set color of the line(red/gree/blue)",  OFFSET(color_line_str), AV_OPT_TYPE_STRING,{ .str = "red" }, CHAR_MIN, CHAR_MAX, FLAGS },
    { "color_text",     "set color of the line(red/gree/blue)",  OFFSET(color_text_str), AV_OPT_TYPE_STRING,{ .str = "green" }, CHAR_MIN, CHAR_MAX, FLAGS },
    { "thickness", "set the box thickness value", OFFSET(thickness), AV_OPT_TYPE_INT,{ .i64 = 1 }, 1, 5 , FLAGS },
    { "line_type", "set the line_type value", OFFSET(line_type), AV_OPT_TYPE_INT,{ .i64 = 4 }, 1, 5 , FLAGS },
    { NULL }
};

AVFILTER_DEFINE_CLASS(ocv_overlay);

static const AVFilterPad avfilter_vf_ocv_overlay_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad avfilter_vf_ocv_overlay_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_ocv_overlay = {
    .name          = "ocv_overlay",
    .description   = NULL_IF_CONFIG_SMALL("Draw rectangle and text using libopencv."),
    .priv_size     = sizeof(OCVOverlayContext),
    .priv_class    = &ocv_overlay_class,
    .query_formats = query_formats,
    .init          = init,
    .inputs        = avfilter_vf_ocv_overlay_inputs,
    .outputs       = avfilter_vf_ocv_overlay_outputs,
};
