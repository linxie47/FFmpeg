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
 * inference base function
 */

#include "formats.h"
#include "internal.h"
#include "avfilter.h"
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"

#include "inference.h"

struct InferenceBaseContext
{
    char *infer_type;
    int   batch_size;
    int   every_nth_frame;
    float threshold;

    DNNModule *module;
    DNNModel  *model;

    DNNModelInfo input_info;
    DNNModelInfo output_info;

    VideoPP vpp;

    InferencePreProcess preprocess;
};

static int fill_dnn_data_from_frame(DNNIOData *data,
                                    const AVFrame *frame,
                                    int batch_idx,
                                    int is_image,
                                    int input_idx)
{
    int channels_nb;
    DNNDataFormat dnn_fmt;
    DNNDataPrecisionType precision;
    enum AVPixelFormat pix_fmt = frame->format;

    switch (pix_fmt)
    {
    case AV_PIX_FMT_GRAY8:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_GRAY_PLANAR;
        channels_nb = 1;
        break;
    case AV_PIX_FMT_BGRA:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_BGR_PACKED;
        channels_nb = 4;
        break;
    case AV_PIX_FMT_BGR24:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_BGR_PACKED;
        channels_nb = 3;
        break;
    default:
        av_log(NULL, AV_LOG_ERROR, "format unsupport!\n");
        return AVERROR(EINVAL);
    };

    data->data           = (void *)frame->data[0];
    data->width          = frame->width;
    data->height         = frame->height;
    data->width_stride   = frame->linesize[0]/channels_nb;
    data->height_stride  = frame->height;
    data->channels       = channels_nb;
    data->data_format    = dnn_fmt;
    data->precision      = precision;
    data->memory_type    = DNN_MEM_HOST;
    data->batch_idx      = batch_idx;
    data->is_image       = is_image;
    data->in_out_idx     = input_idx;

    return 0;
}

static int sw_crop_and_scale(AVFrame *frame,
                             float x0, float y0,
                             float x1, float y1,
                             int out_w, int out_h,
                             enum AVPixelFormat out_format,
                             uint8_t *data[], int stride[])
{
    int err, bufsize;
    struct SwsContext *sws_ctx;
    const AVPixFmtDescriptor *desc;
    int x, y, w, h, hsub, vsub;
    int max_step[4]; ///< max pixel step for each plane, expressed as a number of bytes
    enum AVPixelFormat expect_format = out_format;

    AVFrame *temp = av_frame_alloc();
    if (!temp) {
        err = AVERROR(ENOMEM);
        return err;
    }
    av_frame_ref(temp, frame);

    desc = av_pix_fmt_desc_get(temp->format);
    hsub = desc->log2_chroma_w;
    vsub = desc->log2_chroma_h;
    av_image_fill_max_pixsteps(max_step, NULL, desc);

    /* cropping */
    {
        x = lrintf(x0);
        y = lrintf(y0);
        w = lrintf(x1) - x;
        h = lrintf(y1) - y;

        temp->width  = w;
        temp->height = h;

        temp->data[0] += y * temp->linesize[0];
        temp->data[0] += x * max_step[0];

        for (int i = 1; i < 3; i ++) {
            if (temp->data[i]) {
                temp->data[i] += (y >> vsub) * temp->linesize[i];
                temp->data[i] += (x * max_step[i]) >> hsub;
            }
        }

        /* alpha plane */
        if (temp->data[3]) {
            temp->data[3] += y * temp->linesize[3];
            temp->data[3] += x * max_step[3];
        }
    }

    /* create scaling context */
    sws_ctx = sws_getContext(temp->width, temp->height, temp->format,
                             out_w, out_h, expect_format,
                             SWS_BILINEAR, NULL, NULL, NULL);
    if (!sws_ctx) {
        av_log(NULL, AV_LOG_ERROR, "Create scaling context failed!\n");
        err = AVERROR(EINVAL);
        return err;
    }

    if (!data[0]) {
        bufsize = av_image_alloc(data, stride, out_w, out_h, expect_format, 1);
        if (bufsize < 0)
            return AVERROR(ENOMEM);
    }

    sws_scale(sws_ctx, (const uint8_t * const*)temp->data,
              temp->linesize, 0, temp->height, data, stride);

    av_frame_free(&temp);
    sws_freeContext(sws_ctx);

    return 0;
}

void av_split(char *str, const char *delim, char **array, int *num, int max)
{
    char *p;
    int i = 0;

    if (!str || !delim || !array || !num)
        return;

    p = strtok(str, delim);
    while (p != NULL) {
        array[i++] = p;

        av_assert0 (i < max);

        p = strtok(NULL, delim);
    }
    *num = i;
}

double av_norm(float vec[], size_t num)
{
    size_t i;
    double result = 0.0;

    for (i = 0; i < num; i++)
        result += vec[i] * vec[i];

    return sqrt(result);
}

double av_dot(float vec1[], float vec2[], size_t num)
{
    size_t i;
    double result = 0.0;

    for (i = 0; i < num; i++)
        result += vec1[i] * vec2[i];

    return result;
}

int ff_inference_base_create(AVFilterContext *ctx,
                             InferenceBaseContext **base,
                             InferenceParam *param)
{
    int i, ret;
    InferenceBaseContext *s;
    VideoPP *vpp;
    DNNModelInfo *info;
    DNNModelIntelIEConfig config;

    if (!param)
        return AVERROR(EINVAL);

    s = av_mallocz(sizeof(*s));
    if (!s)
        return AVERROR(ENOMEM);

    // TODO: handle hw ctx

    s->module = ff_get_dnn_module(param->backend_type);
    if (!s->module) {
        av_log(ctx, AV_LOG_ERROR, "could not create DNN backend module\n");
        av_freep(&s);
        return AVERROR(ENOMEM);
    }

    // parameter sanity check
    if (param->batch_size <= 0) param->batch_size = 1;

    config = (DNNModelIntelIEConfig) {
        .model         = param->model_file,
        .labels        = param->labels_file,
        .device        = param->device_type,
        .batch_size    = param->batch_size,
        .cpu_extension = param->cpu_extension,
        .gpu_extension = param->gpu_extension,
    };
    s->model = s->module->load_model_with_config(&config);
    if (!s->model) {
        av_log(ctx, AV_LOG_ERROR, "could not load DNN model\n");
        av_freep(&s);
        return AVERROR(ENOMEM);
    }

#define DNN_ERR_CHECK(ctx) \
    if (ret != DNN_SUCCESS) { \
        av_log(ctx, AV_LOG_ERROR, "Error in '%s' line %d: %d\n", __FUNCTION__, __LINE__, ret); \
        goto fail; \
    }\

    ret = s->model->get_input_info(s->model->model, &s->input_info);
    DNN_ERR_CHECK(ctx);

    ret = s->model->get_output_info(s->model->model, &s->output_info);
    DNN_ERR_CHECK(ctx);

    info = &s->input_info;
    for (i = 0; i < info->numbers; i++) {
        info->layout[i]    = param->input_layout;
        info->precision[i] = param->input_precision;
        info->is_image[i]  = param->input_is_image;
    }
    ret = s->model->set_input_info(s->model->model, info);
    DNN_ERR_CHECK(ctx);

    s->batch_size      = param->batch_size;
    s->every_nth_frame = param->every_nth_frame;
    s->threshold       = param->threshold;
    s->preprocess      = param->preprocess;

    ret = s->model->create_model(s->model->model);
    DNN_ERR_CHECK(ctx);

    vpp = &s->vpp;

    // vpp init
    vpp->swscale        = &sws_scale;
    vpp->crop_and_scale = &sw_crop_and_scale;
    vpp->expect_format  = AV_PIX_FMT_BGR24;

    *base = s;
#undef DNN_ERR_CHECK
    return 0;
fail:
    av_freep(&s);
    return ret;
}

int ff_inference_base_free(InferenceBaseContext **base)
{
    InferenceBaseContext *s = *base;

    if (!s)
        return 0;

    if (s->vpp.device == VPP_DEVICE_SW) {
        for (int i = 0; i < MAX_VPP_NUM; i++) {
            if (s->vpp.frames[i])
                av_frame_free(&s->vpp.frames[i]);
            if (s->vpp.scale_contexts[i])
                sws_freeContext(s->vpp.scale_contexts[i]);
        }
    }

    if (s->module) {
        s->module->free_model(&s->model);
        av_freep(&s->module);
    }

    av_freep(base);
    return 0;
}

int ff_inference_base_submit_frame(InferenceBaseContext *base,
                                   AVFrame *frame,
                                   int input_idx,
                                   int batch_idx)
{
    DNNIOData input = { };
    fill_dnn_data_from_frame(&input, frame, batch_idx, 1, input_idx);
    base->model->set_input(base->model->model, &input);

    return 0;
}

int ff_inference_base_infer(InferenceBaseContext *base)
{
    DNNReturnType dnn_ret;
    dnn_ret = base->module->execute_model(base->model);
    av_assert0(dnn_ret == DNN_SUCCESS);
    return 0;
}

int ff_inference_base_filter_frame(InferenceBaseContext *base, AVFrame *in)
{
    DNNModelInfo *info = &base->input_info;
    DNNReturnType dnn_ret;
    DNNIOData input = { };

    for (int i = 0; i < info->numbers; i++) {
        AVFrame *processed_frame;
        for (int j = 0; j < base->batch_size; j++) {
            if (base->preprocess)
                base->preprocess(base, i, in, &processed_frame);
            fill_dnn_data_from_frame(&input, processed_frame, j, 1, i);
            base->model->set_input(base->model->model, &input);
        }
    }

    dnn_ret = base->module->execute_model(base->model);
    av_assert0(dnn_ret == DNN_SUCCESS);

    return 0;
}

int ff_inference_base_get_infer_result(InferenceBaseContext *base,
                                       int output_index,
                                       InferTensorMeta *metadata)
{
    DNNModelInfo *info = &base->output_info;
    DNNIOData     data = { };
    DNNReturnType ret;

    av_assert0(metadata != NULL);

    // TODO: change to layer name for multiple outputs
    data.in_out_idx = output_index;

    ret = base->model->get_execute_result(base->model->model, &data);
    av_assert0(ret == DNN_SUCCESS);

    //TODO: refine by new interface
    metadata->dim_size  = 3;
    metadata->dims[0]   = info->width[0];
    metadata->dims[1]   = info->height[0];
    metadata->dims[2]   = info->channels[0];
    metadata->layout    = info->layout[0];
    metadata->precision = info->precision[0];

    metadata->data        = data.data;
    metadata->total_bytes = data.size;

    return 0;
}

DNNModelInfo* ff_inference_base_get_input_info(InferenceBaseContext *base)
{
    return &base->input_info;
}

DNNModelInfo* ff_inference_base_get_output_info(InferenceBaseContext *base)
{
    return &base->output_info;
}

VideoPP* ff_inference_base_get_vpp(InferenceBaseContext *base)
{
    return &base->vpp;
}
