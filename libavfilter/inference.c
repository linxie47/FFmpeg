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

#if CONFIG_LIBJSON_C
#include <json-c/json.h>
#endif

#if CONFIG_VAAPI
#define VA_CALL(_FUNC)                                     \
    {                                                      \
        VAStatus _status = _FUNC;                          \
        if (_status != VA_STATUS_SUCCESS)                  \
        {                                                  \
            printf(#_FUNC " failed, sts = %d (%s).\n",     \
                    _status, vaErrorStr(_status));         \
            return AVERROR(EINVAL);                        \
        }                                                  \
    }
#endif

struct _InferenceBaseContext
{
    char *infer_type;
    int   batch_size;

    DNNModule *module;
    DNNModel  *model;

    DNNModelInfo input_info;
    DNNModelInfo output_info;

    VideoPP vpp;

    InferencePreProcess preprocess;
};

static int va_vpp_crop_and_scale(VAAPIVpp *va_vpp, AVFrame *input, Rect *crop_rect,
        int scale_w, int scale_h, uint8_t *data[],  int stride[]);

static int va_vpp_scale(VAAPIVpp *va_vpp, AVFrame *input,
        int scale_w, int scale_h, uint8_t *data[],  int stride[]);

static void infer_labels_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    LabelsArray *labels = (LabelsArray *)data;

    for (i = 0; i < labels->num; i++)
        av_freep(&labels->label[i]);

    av_free(labels->label);

    av_free(data);
}

// helper functions
static void infer_labels_dump(uint8_t *data)
{
    int i;
    LabelsArray *labels = (LabelsArray *)data;
    printf("labels: ");
    for (i = 0; i < labels->num; i++)
        printf("%s ", labels->label[i]);
    printf("\n");
}

int ff_get_file_size(FILE *fp)
{
    int file_size, current_pos;

    if (!fp)
        return -1;

    current_pos = ftell(fp);

    if (fseek(fp, 0, SEEK_END)) {
        fprintf(stderr, "Couldn't seek to the end of feature file.\n");
        return -1;
    }

    file_size = ftell(fp);

    fseek(fp, current_pos, SEEK_SET);

    return file_size;
}


static int fill_dnn_data_from_frame(DNNIOData *data,
                                    const AVFrame *frame,
                                    int batch_idx,
                                    int is_image,
                                    int input_idx)
{
    int i, channels_nb;
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
    case AV_PIX_FMT_BGR0:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_BGR_PACKED;
        channels_nb = 4;
        break;
    case AV_PIX_FMT_BGR24:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_BGR_PACKED;
        channels_nb = 3;
        break;
    case AV_PIX_FMT_RGBP:
        precision = DNN_DATA_PRECISION_U8;
        dnn_fmt = DNN_DATA_RGB_PLANAR;
        channels_nb = 3;
        break;
    default:
        av_log(NULL, AV_LOG_ERROR, "format unsupport!\n");
        return AVERROR(EINVAL);
    };

    for (i = 0; i < NUM_DATA_POINTS; i++) {
        data->data[i]     = frame->data[i];
        data->linesize[i] = frame->linesize[i];
    }
    data->width          = frame->width;
    data->height         = frame->height;
    data->channels       = channels_nb;
    data->data_format    = dnn_fmt;
    data->precision      = precision;
    data->memory_type    = DNN_MEM_HOST;
    data->batch_idx      = batch_idx;
    data->is_image       = is_image;
    data->in_out_idx     = input_idx;

    return 0;
}

static int sw_crop_and_scale(AVFrame *frame, Rect *crop_rect,
                             int out_w,      int out_h,
                             enum AVPixelFormat out_format,
                             uint8_t *data[], int stride[])
{
    int ret = 0;
    AVFrame *temp = NULL;
    struct SwsContext *sws_ctx = NULL;
    const AVPixFmtDescriptor *desc;
    int x, y, w, h, hsub, vsub, bufsize;
    int max_step[4]; ///< max pixel step for each plane, expressed as a number of bytes
    enum AVPixelFormat expect_format = out_format;

    if (!crop_rect)
        return AVERROR(EINVAL);

    temp = av_frame_alloc();
    if (!temp)
        return AVERROR(ENOMEM);

    av_frame_ref(temp, frame);

    desc = av_pix_fmt_desc_get(temp->format);
    if (!desc) {
        ret = AVERROR(EINVAL);
        goto exit;
    }
    hsub = desc->log2_chroma_w;
    vsub = desc->log2_chroma_h;
    av_image_fill_max_pixsteps(max_step, NULL, desc);

    /* cropping */
    {
        x = lrintf(crop_rect->x0);
        y = lrintf(crop_rect->y0);
        x = FFMAX(x, 0);
        y = FFMAX(y, 0);
        if (x >= frame->width || y >= frame->height) {
            av_log(NULL, AV_LOG_ERROR, "Incorrect crop rect x:%d y:%d.\n", x, y);
            ret = AVERROR(EINVAL);
            goto exit;
        }

        w = lrintf(crop_rect->x1) - x;
        h = lrintf(crop_rect->y1) - y;
        w = FFMIN(w, frame->width - x);
        h = FFMIN(h, frame->height - y);
        if (w <= 0 || h <= 0) {
            av_log(NULL, AV_LOG_ERROR, "Incorrect crop rect w:%d h:%d.\n", w, h);
            ret = AVERROR(EINVAL);
            goto exit;
        }

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
        ret = AVERROR(EINVAL);
        goto exit;
    }

    if (!data[0]) {
        bufsize = av_image_alloc(data, stride, out_w, out_h, expect_format, 1);
        if (bufsize < 0) {
            ret = AVERROR(ENOMEM);
            goto exit;
        }
    }

    sws_scale(sws_ctx, (const uint8_t * const*)temp->data,
              temp->linesize, 0, temp->height, data, stride);
exit:
    av_frame_free(&temp);
    sws_freeContext(sws_ctx);
    return ret;
}

void av_split(char *str, const char *delim, char **array, int *num, int max)
{
    char *p;
    int i = 0;

    if (!str || !delim || !array || !num)
        return;

    while (p = strtok(str, delim)) {
        int j = 0;
        char *s;
        size_t end;

        /* remove head blanks */
        while (p[j] == '\n' || p[j] == ' ')
            j++;

        if (!p[j]) continue;

        /* remove tail blanks */
        s   = p + j;
        end = strlen(s) - 1;
        while (s[end] == '\n' || s[end] == ' ')
            s[end--] = '\0';

        array[i++] = s;
        av_assert0 (i < max);

        /* string is cached */
        str = NULL;
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
        av_freep(&s->module);
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
    for (i = 0; i < info->number; i++) {
        info->layout[i]    = param->input_layout;
        info->precision[i] = param->input_precision;
        info->is_image[i]  = param->input_is_image;
    }
    ret = s->model->set_input_info(s->model->model, info);
    DNN_ERR_CHECK(ctx);

    s->batch_size      = param->batch_size;
    s->preprocess      = param->preprocess;

    ret = s->model->create_model(s->model->model);
    DNN_ERR_CHECK(ctx);

    vpp = &s->vpp;

    // vpp init
    vpp->sw_vpp = av_mallocz(sizeof(*vpp->sw_vpp));
    if (!vpp->sw_vpp)
        goto fail;

    vpp->expect_format          = AV_PIX_FMT_BGR24;
    vpp->sw_vpp->scale          = &sws_scale;
    vpp->sw_vpp->crop_and_scale = &sw_crop_and_scale;

    *base = s;
#undef DNN_ERR_CHECK
    return 0;
fail:
    s->module->free_model(&s->model);
    av_freep(&s->module);
    av_freep(&s);
    return ret;
}

int ff_inference_base_free(InferenceBaseContext **base)
{
    InferenceBaseContext *s = *base;

    if (!s)
        return 0;

    // VPP clean up
    for (int i = 0; i < MAX_VPP_NUM; i++) {
        if (s->vpp.frames[i])
            av_frame_free(&s->vpp.frames[i]);
        if (s->vpp.sw_vpp->scale_contexts[i])
            sws_freeContext(s->vpp.sw_vpp->scale_contexts[i]);
    }
    av_freep(&s->vpp.sw_vpp);

#if CONFIG_VAAPI
    if (s->vpp.va_vpp) {
        va_vpp_device_free(s->vpp.va_vpp);
        av_freep(&s->vpp.va_vpp);
    }
#endif

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
#if CONFIG_VAAPI
    if (base->vpp.va_vpp)
        va_vpp_surface_release(base->vpp.va_vpp);
#endif

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

    for (int i = 0; i < info->number; i++) {
        AVFrame *processed_frame = NULL;
        for (int j = 0; j < base->batch_size; j++) {
            if (base->preprocess) {
                if (base->preprocess(base, i, in, &processed_frame) < 0)
                    return AVERROR(EINVAL);
            }

            if (!processed_frame) return -1;

            fill_dnn_data_from_frame(&input, processed_frame, j, 1, i);
            base->model->set_input(base->model->model, &input);
#if CONFIG_VAAPI
            if (base->vpp.va_vpp)
                va_vpp_surface_release(base->vpp.va_vpp);
#endif
        }
    }

    dnn_ret = base->module->execute_model(base->model);
    av_assert0(dnn_ret == DNN_SUCCESS);

    return 0;
}

int ff_inference_base_get_infer_result(InferenceBaseContext *base,
                                       int id,
                                       InferTensorMeta *metadata)
{
    DNNModelInfo *info = &base->output_info;
    DNNIOData     data = { };
    DNNReturnType ret;

    av_assert0(metadata != NULL);

    av_assert0(id < DNN_INPUT_OUTPUT_NUM);

    // TODO: change to layer name for multiple outputs
    data.in_out_idx = id;

    ret = base->model->get_execute_result(base->model->model, &data);
    av_assert0(ret == DNN_SUCCESS);

    metadata->dim_size  = 4;
    memcpy(&metadata->dims[0], &info->dims[id][0],
            metadata->dim_size * sizeof(metadata->dims[0]));

    metadata->layout    = info->layout[id];
    metadata->precision = info->precision[id];

    metadata->data        = data.data[0];
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

void ff_inference_dump_model_info(void *ctx, DNNModelInfo *info)
{
    int i;
    for (i = 0; i < info->number; i++) {
        size_t *p = &info->dims[i][0];
        av_log(ctx, AV_LOG_DEBUG, "Info id:%d layer\"%-16s\" "
               "batch size:%d - dim: %3lu %3lu %3lu %3lu - img:%d pre:%d layout:%d\n",
               i, info->layer_name[i],
               info->batch_size, p[0], p[1], p[2], p[3],
               info->is_image[i], info->precision[i], info->layout[i]);
    }
}

/*
 * VAAPI VPP APIs
 */

#if CONFIG_VAAPI
static int ff_vaapi_vpp_colour_standard(enum AVColorSpace av_cs)
{
    switch(av_cs) {
#define CS(av, va) case AVCOL_SPC_ ## av: return VAProcColorStandard ## va;
        CS(BT709,     BT709);
        CS(BT470BG,   BT601);
        // WORKAROUND: vaapi driver doesn't support all color space
        CS(SMPTE170M, None); //SMPTE170M);
        CS(SMPTE240M, None); //SMPTE240M);
#undef CS
    default:
        return VAProcColorStandardNone;
    }
}

int va_vpp_device_create(VAAPIVpp *va_vpp, AVFilterLink *inlink)
{
    AVFilterContext *avctx = inlink->dst;
    VADisplay display = NULL;
    VAImageFormat *image_list = NULL;
    VAStatus vas;
    int err, image_count;
    AVBufferRef *device_ref = NULL;
    AVHWFramesContext *hw_frames_ctx;

    hw_frames_ctx = (AVHWFramesContext *)inlink->hw_frames_ctx->data;
    av_assert0(hw_frames_ctx);

    device_ref = av_buffer_ref(hw_frames_ctx->device_ref);
    if (!device_ref) {
        av_log(avctx, AV_LOG_ERROR, "A device reference create failed.\n");
        return AVERROR(ENOMEM);
    }

    va_vpp->hwctx         = ((AVHWDeviceContext *)device_ref->data)->hwctx;
    va_vpp->hw_frames_ref = inlink->hw_frames_ctx;

    av_buffer_unref(&device_ref);

    display = va_vpp->hwctx->display;

    image_count = vaMaxNumImageFormats(display);
    if (image_count <= 0) {
        err = AVERROR(EIO);
        goto fail;
    }
    image_list = av_malloc(image_count * sizeof(*image_list));
    if (!image_list) {
        err = AVERROR(ENOMEM);
        goto fail;
    }
    vas = vaQueryImageFormats(display, image_list, &image_count);
    if (vas != VA_STATUS_SUCCESS) {
        err = AVERROR(EIO);
        goto fail;
    }

    va_vpp->format_list = image_list;
    va_vpp->nb_formats  = image_count;
    va_vpp->va_config   = VA_INVALID_ID;
    va_vpp->va_context  = VA_INVALID_ID;
    va_vpp->va_surface  = VA_INVALID_ID;

    va_vpp->scale          = &va_vpp_scale;
    va_vpp->crop_and_scale = &va_vpp_crop_and_scale;

    return VA_STATUS_SUCCESS;
fail:
    if (image_list)
        av_free(image_list);
    return err;
}

int va_vpp_device_free(VAAPIVpp *va_vpp)
{
    VAStatus vas;

    if (!va_vpp)
        return 0;

    if (va_vpp->va_surface != VA_INVALID_ID) {
        vas = vaDestroySurfaces(va_vpp->hwctx->display, &va_vpp->va_surface, 1);
        if (vas != VA_STATUS_SUCCESS) {
            av_log(NULL, AV_LOG_ERROR, "Failed to destroy surface %#x: "
                    "%d (%s).\n", va_vpp->va_surface, vas, vaErrorStr(vas));
        }
    }

    if (va_vpp->va_context != VA_INVALID_ID) {
        vaDestroyContext(va_vpp->hwctx->display, va_vpp->va_context);
        va_vpp->va_context = VA_INVALID_ID;
    }

    if (va_vpp->va_config != VA_INVALID_ID) {
        vaDestroyConfig(va_vpp->hwctx->display, va_vpp->va_config);
        va_vpp->va_config = VA_INVALID_ID;
    }

    av_free(va_vpp->format_list);

    return VA_STATUS_SUCCESS;
}

int va_vpp_surface_alloc(VAAPIVpp *va_vpp, size_t width, size_t height, const char *format)
{
    int i, rt_format, fourcc;
    VASurfaceAttrib surface_attrib;
    enum AVPixelFormat av_format;

    if (!va_vpp)
        return -1;

    if (format == NULL || strstr(format, "bgrx")) {
        fourcc = VA_FOURCC_BGRX; rt_format = VA_RT_FORMAT_RGB32; av_format = AV_PIX_FMT_BGR0;
    } else if (strstr(format, "rgbp")) {
        fourcc = VA_FOURCC_RGBP; rt_format = VA_RT_FORMAT_RGBP;  av_format = AV_PIX_FMT_RGBP;
    } else
        return -1;

    surface_attrib.type          = VASurfaceAttribPixelFormat;
    surface_attrib.flags         = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type    = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = fourcc;

    VA_CALL(vaCreateConfig(va_vpp->hwctx->display, VAProfileNone,
                           VAEntrypointVideoProc, 0, 0, &va_vpp->va_config));

    VA_CALL(vaCreateSurfaces(va_vpp->hwctx->display, rt_format, width, height,
                             &va_vpp->va_surface, 1, &surface_attrib, 1));

    VA_CALL(vaCreateContext(va_vpp->hwctx->display, va_vpp->va_config,
                            width, height, VA_PROGRESSIVE,
                            &va_vpp->va_surface, 1, &va_vpp->va_context));

    for (i = 0; i < va_vpp->nb_formats; i++) {
        if (va_vpp->format_list[i].fourcc == fourcc) {
            va_vpp->va_format_selected = va_vpp->format_list[i];
            break;
        }
    }

    va_vpp->av_format = av_format;

    return VA_STATUS_SUCCESS;
}

/* release mapped memory */
int va_vpp_surface_release(VAAPIVpp *va_vpp)
{
    VA_CALL(vaUnmapBuffer(va_vpp->hwctx->display, va_vpp->va_image.buf));
    VA_CALL(vaDestroyImage(va_vpp->hwctx->display, va_vpp->va_image.image_id));

    return VA_STATUS_SUCCESS;
}

/* HW scale and map to system memory */
static int va_vpp_scale(VAAPIVpp *va_vpp, AVFrame *input,
                        int scale_w,      int scale_h,
                        uint8_t *data[],  int stride[])
{
    return va_vpp_crop_and_scale(va_vpp, input, NULL, scale_w, scale_h, data, stride);
}

static int va_vpp_crop_and_scale(VAAPIVpp *va_vpp,
                                 AVFrame *input,   Rect *crop_rect,
                                 int scale_w,      int scale_h,
                                 uint8_t *data[],  int stride[])
{
    int i;
    void *address = NULL;
    VAImage   *va_image_ptr;
    VABufferID params_id;
    VASurfaceID input_surface, output_surface;
    VAProcPipelineParameterBuffer params;
    VARectangle input_region;

    input_surface = (VASurfaceID)(uintptr_t)input->data[3];
    av_log(NULL, AV_LOG_DEBUG, "Using surface %#x for scale input.\n",
           input_surface);

    output_surface = va_vpp->va_surface;

    if (crop_rect == NULL) {
        input_region = (VARectangle) {
            .x      = input->crop_left,
            .y      = input->crop_top,
            .width  = input->width -
                     (input->crop_left + input->crop_right),
            .height = input->height -
                     (input->crop_top + input->crop_bottom),
        };
    } else {
        int _x = lrintf(crop_rect->x0);
        int _y = lrintf(crop_rect->y0);
        _x = FFMAX(_x, 0);
        _y = FFMAX(_y, 0);
        if (_x >= input->width  || _y >= input->height) {
            av_log(NULL, AV_LOG_ERROR, "Incorrect crop rect!\n");
            return AVERROR(EINVAL);
        }
        input_region = (VARectangle) {
            .x      = _x,
            .y      = _y,
            .width  = FFMIN(lrintf(crop_rect->x1) - _x, input->width - _x),
            .height = FFMIN(lrintf(crop_rect->y1) - _y, input->height - _y),
        };
    }

    memset(&params, 0, sizeof(params));

    params.surface = input_surface;
    params.surface_region = &input_region;
    params.surface_color_standard =
        ff_vaapi_vpp_colour_standard(input->colorspace);

    params.output_region = 0;
    params.output_background_color = 0xff000000;
    params.output_color_standard = params.surface_color_standard;

    params.pipeline_flags = 0;
    params.filter_flags = VA_FILTER_SCALING_HQ;

    VA_CALL(vaBeginPicture(va_vpp->hwctx->display, va_vpp->va_context, output_surface));

    VA_CALL(vaCreateBuffer(va_vpp->hwctx->display, va_vpp->va_context,
                           VAProcPipelineParameterBufferType,
                           sizeof(params), 1, &params, &params_id));

    VA_CALL(vaRenderPicture(va_vpp->hwctx->display, va_vpp->va_context,
                            &params_id, 1));

    VA_CALL(vaEndPicture(va_vpp->hwctx->display, va_vpp->va_context));

    VA_CALL(vaDestroyBuffer(va_vpp->hwctx->display, params_id));

    VA_CALL(vaSyncSurface(va_vpp->hwctx->display, output_surface));

    // map surface to system memory
    va_image_ptr = &va_vpp->va_image;

    VA_CALL(vaCreateImage(va_vpp->hwctx->display, &va_vpp->va_format_selected,
                          scale_w, scale_h, va_image_ptr));

    VA_CALL(vaGetImage(va_vpp->hwctx->display, output_surface,
                       0, 0, scale_w, scale_h,
                       va_image_ptr->image_id));

    VA_CALL(vaMapBuffer(va_vpp->hwctx->display, va_image_ptr->buf, &address));

    for (i = 0; i < va_image_ptr->num_planes; i++) {
        data[i]   = (uint8_t *)address + va_image_ptr->offsets[i];
        stride[i] = va_image_ptr->pitches[i];
    }

    return VA_STATUS_SUCCESS;
}
#endif

#if CONFIG_LIBJSON_C
/*
 * model proc parsing functions using JSON-c
 */
void *ff_read_model_proc(const char *path)
{
    int n, file_size;
    json_object *proc_config = NULL;
    uint8_t *proc_json = NULL;
    json_tokener *tok = NULL;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "File open error:%s\n", path);
        return NULL;
    }

    file_size = ff_get_file_size(fp);

    proc_json = av_mallocz(file_size + 1);
    if (!proc_json)
        goto end;

    n = fread(proc_json, file_size, 1, fp);

    UNUSED(n);

    tok = json_tokener_new();
    proc_config = json_tokener_parse_ex(tok, proc_json, file_size);
    if (proc_config == NULL) {
        enum json_tokener_error jerr;
        jerr = json_tokener_get_error(tok);
        fprintf(stderr, "Error before: %s\n", json_tokener_error_desc(jerr));
        goto end;
    }

end:
    if (proc_json)
        av_freep(&proc_json);
    if(tok)
        json_tokener_free(tok);
    fclose(fp);
    return proc_config;
}

void ff_load_default_model_proc(ModelInputPreproc *preproc, ModelOutputPostproc *postproc)
{
    if (preproc) {
        /*
         * format is a little tricky, an ideal input format for IE is BGR planer
         * however, neither soft csc nor hardware vpp could support that format.
         * Here, we set a close soft format. The actual one coverted before sent
         * to IE will be decided by user config and hardware vpp used or not.
         */
        preproc->color_format = AV_PIX_FMT_BGR24;
        preproc->layer_name   = NULL;
    }

    if (postproc) {
        // do nothing
    }
}

int ff_parse_input_preproc(const void *json, ModelInputPreproc *m_preproc)
{
    json_object *jvalue, *preproc, *color, *layer, *object_class;
    int ret;

    ret = json_object_object_get_ex((json_object *)json, "input_preproc", &preproc);
    if (!ret) {
        av_log(NULL, AV_LOG_DEBUG, "No input_preproc.\n");
        return 0;
    }

    // not support multiple inputs yet
    av_assert0(json_object_array_length(preproc) <= 1);

    jvalue = json_object_array_get_idx(preproc, 0);

    ret = json_object_object_get_ex(jvalue, "color_format", &color);
    if (ret) {
        if (json_object_get_string(color) == NULL)
            return -1;

        av_log(NULL, AV_LOG_INFO, "Color Format:\"%s\"\n", json_object_get_string(color));

        if (!strcmp(json_object_get_string(color), "BGR"))
            m_preproc->color_format = AV_PIX_FMT_BGR24;
        else if (!strcmp(json_object_get_string(color), "RGB"))
            m_preproc->color_format = AV_PIX_FMT_RGB24;
        else
            return -1;
    }

    ret = json_object_object_get_ex(jvalue, "object_class", &object_class);
    if (ret) {
        if (json_object_get_string(object_class) == NULL)
            return -1;

        av_log(NULL, AV_LOG_INFO, "Object_class:\"%s\"\n", json_object_get_string(object_class));

        m_preproc->object_class = (char *)json_object_get_string(object_class);
    }

    ret = json_object_object_get_ex(jvalue, "layer_name", &layer);
    UNUSED(layer);

    return 0;
}

// For detection, we now care labels only.
// Layer name and type can be got from output blob.
int ff_parse_output_postproc(const void *json, ModelOutputPostproc *m_postproc)
{
    json_object *jvalue, *postproc;
    json_object *attribute, *converter, *labels, *layer, *method, *threshold;
    json_object *tensor_to_text_scale, *tensor_to_text_precision;
    int ret;
    size_t jarraylen;

    ret = json_object_object_get_ex((json_object *)json, "output_postproc", &postproc);
    if (!ret) {
        av_log(NULL, AV_LOG_DEBUG, "No output_postproc.\n");
        return 0;
    }

    jarraylen = json_object_array_length(postproc);
    av_assert0(jarraylen <= MAX_MODEL_OUTPUT);

    for(int i = 0; i < jarraylen; i++){
        OutputPostproc *proc = &m_postproc->procs[i];
        jvalue = json_object_array_get_idx(postproc, i);

#define FETCH_STRING(var, name)                                           \
        do { ret = json_object_object_get_ex(jvalue, #name, &var);        \
            if (ret) proc->name = (char *)json_object_get_string(var);    \
        } while(0)
#define FETCH_DOUBLE(var, name)                                           \
        do { ret = json_object_object_get_ex(jvalue, #name, &var);        \
            if (ret) proc->name = (double)json_object_get_double(var);    \
        } while(0)
#define FETCH_INTEGER(var, name)                                          \
        do { ret = json_object_object_get_ex(jvalue, #name, &var);        \
            if (ret) proc->name = (int)json_object_get_int(var);          \
        } while(0)

        FETCH_STRING(layer, layer_name);
        FETCH_STRING(method, method);
        FETCH_STRING(attribute, attribute_name);
        FETCH_STRING(converter, converter);

        FETCH_DOUBLE(threshold, threshold);
        FETCH_DOUBLE(tensor_to_text_scale, tensor_to_text_scale);

        FETCH_INTEGER(tensor_to_text_precision, tensor_to_text_precision);

        // handle labels
        ret = json_object_object_get_ex(jvalue, "labels", &labels);
        if (ret) {
            json_object *label;
            size_t labels_num = json_object_array_length(labels);

            if (labels_num > 0) {
                AVBufferRef *ref    = NULL;
                LabelsArray *larray = av_mallocz(sizeof(*larray));

                if (!larray)
                    return AVERROR(ENOMEM);

                for(int i = 0; i < labels_num; i++){
                    char *l;
                    label = json_object_array_get_idx(labels, i);
                    l = av_strdup(json_object_get_string(label));
                    av_dynarray_add(&larray->label, &larray->num, l);
                }

                ref = av_buffer_create((uint8_t *)larray, sizeof(*larray),
                        &infer_labels_buffer_free, NULL, 0);

                proc->labels = ref;

                if(ref)
                    infer_labels_dump(ref->data);
            }
        }
    }

#undef FETCH_STRING
#undef FETCH_DOUBLE
#undef FETCH_INTEGER

    return 0;
}

void ff_release_model_proc(const void *json,
        ModelInputPreproc *preproc, ModelOutputPostproc *postproc)
{
    size_t index = 0;

    if (!json) return;

    if (postproc) {
        for (index = 0; index < MAX_MODEL_OUTPUT; index++) {
            if (postproc->procs[index].labels)
                av_buffer_unref(&postproc->procs[index].labels);
        }
    }

    json_object_put((json_object *)json);
}
#endif
