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

#define MAX_MODEL_NUM 8

typedef int (*ClassifyProcess)(AVFilterContext*, int, int, int,
                               InferTensorMeta*, InferClassificationMeta*);

typedef struct InferenceClassifyContext {
    const AVClass *class;

    InferenceBaseContext *infer_bases[MAX_MODEL_NUM];

    char  *model_file;
    char  *model_proc;
    char  *vpp_format;

    int    loaded_num;
    int    backend_type;
    int    device_type;

    int    batch_size;
    int    frame_number;
    int    every_nth_frame;

    ClassifyProcess post_process[MAX_MODEL_NUM];

    void *proc_config[MAX_MODEL_NUM];
    ModelInputPreproc   model_preproc[MAX_MODEL_NUM];
    ModelOutputPostproc model_postproc[MAX_MODEL_NUM];
} InferenceClassifyContext;

static void infer_classify_metadata_buffer_free(void *opaque, uint8_t *data)
{
    int i;
    InferClassificationMeta *meta = (InferClassificationMeta *)data;
    ClassifyArray *classes        = meta->c_array;

    if (classes) {
        for (i = 0; i < classes->num; i++) {
            InferClassification *c = classes->classifications[i];
            av_buffer_unref(&c->label_buf);
            av_buffer_unref(&c->tensor_buf);
            av_freep(&c);
        }
        av_free(classes->classifications);
        av_freep(&classes);
    }

    av_free(data);
}

static av_cold void dump_softmax(AVFilterContext *ctx, char *name, int label_id,
                                 float conf, AVBufferRef *label_buf)
{
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(ctx, AV_LOG_DEBUG, "CLASSIFY META - Label id:%d %s:%s Conf:%f\n",
           label_id, name, array->label[label_id], conf);
}

static av_cold void dump_tensor_value(AVFilterContext *ctx, char *name, float value)
{
    av_log(ctx, AV_LOG_DEBUG, "CLASSIFY META - %s:%1.2f\n", name, value);
}

static void find_max_element_index(const float *array, int len,
                                   int *index, float *value)
{
    int i;
    *index = 0;
    *value = array[0];
    for (i = 1; i < len; i++) {
        if (array[i] > *value) {
            *index = i;
            *value = array[i];
        }
    }
}

static int attributes_to_text(AVFilterContext *ctx,
                              int detect_id,
                              OutputPostproc *proc,
                              InferTensorMeta *meta,
                              InferClassificationMeta *c_meta)
{
    InferClassification *classify;
    uint32_t method_max, method_compound, method_index;
    const float *data = (const float *)meta->data;

    method_max      = !strcmp(proc->method, "max");
    method_compound = !strcmp(proc->method, "compound");
    method_index    = !strcmp(proc->method, "index");

    if (!data) return -1;

    classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    if (method_max) {
        int    index;
        float  confidence;
        size_t n = meta->dims[1];

        find_max_element_index(data, n, &index, &confidence);

        classify->detect_id  = detect_id;
        classify->name       = proc->attribute_name;
        classify->label_id   = index;
        classify->confidence = confidence;
        classify->label_buf  = av_buffer_ref(proc->labels);

        if (classify->label_buf) {
            dump_softmax(ctx, classify->name, classify->label_id,
                         classify->confidence,classify->label_buf);
        }
    } else if (method_compound) {
        int i;
        double threshold  = 0.5;
        float  confidence = 0;
        char attributes[4096] = {};
        LabelsArray *array;

        if (proc->threshold != 0)
            threshold = proc->threshold;

        array = (LabelsArray *)proc->labels->data;
        for (i = 0; i < array->num; i++) {
            if (data[i] >= threshold)
                strncat(attributes, array->label[i], (strlen(array->label[i]) + 1));
            if (data[i] > confidence)
                confidence = data[i];
        }

        classify->name = proc->attribute_name;
        classify->confidence = confidence;

        av_log(ctx, AV_LOG_DEBUG, "Attributes: %s\n", attributes);
        // TODO: to add into side data
        av_free(classify);
        return 0;
    } else if (method_index) {
        int i;
        char attributes[1024] = {};
        LabelsArray *array;

        array = (LabelsArray *)proc->labels->data;
        for (i = 0; i < array->num; i++) {
            int value = data[i];
            if (value < 0 || value >= array->num)
                break;
            strncat(attributes, array->label[value], (strlen(array->label[value]) + 1));
        }

        classify->name = proc->attribute_name;

        av_log(ctx, AV_LOG_DEBUG, "Attributes: %s\n", attributes);
        // TODO: to add into side data
        av_free(classify);
        return 0;
    }

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);
    return 0;
}

static int tensor_to_text(AVFilterContext *ctx,
                          int detect_id,
                          OutputPostproc *proc,
                          InferTensorMeta *meta,
                          InferClassificationMeta *c_meta)
{
    InferClassification *classify;
    const float *data = (const float *)meta->data;
    double scale = 1.0;

    if (!data) return -1;

    classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    if (proc->tensor2text_scale != 0)
        scale = proc->tensor2text_scale;

    classify->detect_id = detect_id;
    classify->name      = proc->attribute_name;
    classify->value     = *data * scale;

    dump_tensor_value(ctx, classify->name, classify->value);

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);
    return 0;
}

static int default_postprocess(AVFilterContext *ctx,
                               int detect_id,
                               int result_id,
                               int model_id,
                               InferTensorMeta *meta,
                               InferClassificationMeta *c_meta)
{
    InferenceClassifyContext *s = ctx->priv;
    InferenceBaseContext *base  = s->infer_bases[model_id];
    DNNModelInfo *info = ff_inference_base_get_output_info(base);
    InferClassification *classify;

    if (!meta->data) return -1;

    classify = av_mallocz(sizeof(*classify));
    if (!classify)
        return AVERROR(ENOMEM);

    classify->detect_id  = detect_id;
    classify->layer_name = info->layer_name[result_id];
    classify->model      = s->model_file;
    classify->name       = (char *)"default";

    classify->tensor_buf = av_buffer_alloc(meta->total_bytes);
    if (!classify->tensor_buf) {
        av_free(classify);
        return AVERROR(ENOMEM);
    }

    if (meta->total_bytes > 0)
        memcpy(classify->tensor_buf->data, meta->data, meta->total_bytes);

    av_dynarray_add(&c_meta->c_array->classifications, &c_meta->c_array->num, classify);

    av_log(ctx, AV_LOG_DEBUG, "default output[%s] size: %zu\n", classify->layer_name, meta->total_bytes);
    return 0;
}

static int commmon_postprocess(AVFilterContext *ctx,
                               int detect_id,
                               int result_id,
                               int model_id,
                               InferTensorMeta *meta,
                               InferClassificationMeta *c_meta)
{
    int proc_id;
    InferenceClassifyContext *s = ctx->priv;
    InferenceBaseContext *base  = s->infer_bases[model_id];

    OutputPostproc *proc;
    DNNModelInfo *info = ff_inference_base_get_output_info(base);

    // search model postproc
    for (proc_id = 0; proc_id < MAX_MODEL_OUTPUT; proc_id++) {
        char *proc_layer_name = s->model_postproc[model_id].procs[proc_id].layer_name;

        // skip this output process
        if (!proc_layer_name)
            continue;

        if (!strcmp(info->layer_name[result_id], proc_layer_name))
            break;
    }

    if (proc_id == MAX_MODEL_OUTPUT) {
        av_log(ctx, AV_LOG_DEBUG, "Could not find proc:%s\n", info->layer_name[result_id]);
        return 0;
    }

    proc = &s->model_postproc[model_id].procs[proc_id];

    if (proc->converter == NULL)
        return default_postprocess(ctx, detect_id, result_id, model_id, meta, c_meta);

    if (!strcmp(proc->converter, "tensor_to_label"))
        return attributes_to_text(ctx, detect_id, proc, meta, c_meta);

    if (!strcmp(proc->converter, "tensor_to_text"))
        return tensor_to_text(ctx, detect_id, proc, meta, c_meta);

    return 0;
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

static av_cold int classify_init(AVFilterContext *ctx)
{
    InferenceClassifyContext *s = ctx->priv;
    int i, ret;
    int model_num = 0, model_proc_num = 0;
    const int max_num = MAX_MODEL_NUM;
    char *models[MAX_MODEL_NUM] = { };
    char *models_proc[MAX_MODEL_NUM] = { };
    InferenceParam p = {};

    av_assert0(s->model_file);

    av_split(s->model_file, "&", models, &model_num, max_num);
    for (i = 0; i < model_num; i++)
        av_log(ctx, AV_LOG_INFO, "model[%d]:%s\n", i, models[i]);

    av_split(s->model_proc, "&", models_proc, &model_proc_num, max_num);
    for (i = 0; i < model_proc_num; i++)
        av_log(ctx, AV_LOG_INFO, "proc[%d]:%s\n", i, models_proc[i]);

    av_assert0(s->backend_type == DNN_INTEL_IE);

    p.backend_type    = s->backend_type;
    p.device_type     = s->device_type;
    p.batch_size      = s->batch_size;
    p.input_precision = DNN_DATA_PRECISION_U8;
    p.input_layout    = DNN_DATA_LAYOUT_NCHW;
    p.input_is_image  = 1;

    for (i = 0; i < model_num; i++) {
        void *proc;
        InferenceBaseContext *base = NULL;

        p.model_file = models[i];
        ret = ff_inference_base_create(ctx, &base, &p);
        if (ret < 0) {
            av_log(ctx, AV_LOG_ERROR, "Could not create inference\n");
            return ret;
        }

        s->infer_bases[i] = base;

        ff_load_default_model_proc(&s->model_preproc[i], &s->model_postproc[i]);

        if (!models_proc[i])
            continue;

        proc = ff_read_model_proc(models_proc[i]);
        if (!proc) {
            av_log(ctx, AV_LOG_ERROR, "Could not read proc config file:"
                    "%s\n", models_proc[i]);
            ret = AVERROR(EIO);
            goto fail;
        }

        if (ff_parse_input_preproc(proc, &s->model_preproc[i]) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Parse input preproc error.\n");
            ret = AVERROR(EIO);
            goto fail;
        }

        if (ff_parse_output_postproc(proc, &s->model_postproc[i]) < 0) {
            av_log(ctx, AV_LOG_ERROR, "Parse output postproc error.\n");
            ret = AVERROR(EIO);
            goto fail;
        }

        s->proc_config[i] = proc;
    }
    s->loaded_num = model_num;

    for (i = 0; i < model_num; i++) {
        if (!models_proc[i])
            s->post_process[i] = &default_postprocess;
        else
            s->post_process[i] = &commmon_postprocess;
    }

    return 0;

fail:
    for (i = 0; i < model_num; i++) {
        ff_inference_base_free(&s->infer_bases[i]);
    }

    return ret;
}

static av_cold void classify_uninit(AVFilterContext *ctx)
{
    int i;
    InferenceClassifyContext *s = ctx->priv;

    for (i = 0; i < s->loaded_num; i++) {
        ff_inference_base_free(&s->infer_bases[i]);
        ff_release_model_proc(s->proc_config[i], &s->model_preproc[i], &s->model_postproc[i]);
    }
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    int i, ret = 0;
    AVFilterContext *ctx        = inlink->dst;
    InferenceClassifyContext *s = ctx->priv;
    AVFilterLink *outlink       = inlink->dst->outputs[0];
    AVBufferRef             *ref;
    AVFrameSideData         *sd, *new_sd;
    BBoxesArray             *boxes;
    InferDetectionMeta      *d_meta;
    ClassifyArray           *c_array = NULL;
    InferClassificationMeta *c_meta  = NULL;

    if (s->frame_number % s->every_nth_frame != 0)
        goto done;

    sd = av_frame_get_side_data(in, AV_FRAME_DATA_INFERENCE_DETECTION);
    if (!sd)
        goto done;

    d_meta = (InferDetectionMeta *)sd->data;
    if (!d_meta)
        goto done;

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
            ModelInputPreproc *preproc = &s->model_preproc[j];

            VideoPP *vpp        = ff_inference_base_get_vpp(base);
            AVFrame *tmp        = vpp->frames[0];
            DNNModelInfo *iinfo = ff_inference_base_get_input_info(base);
            DNNModelInfo *oinfo = ff_inference_base_get_output_info(base);
            int scale_width     = iinfo->dims[0][0];
            int scale_height    = iinfo->dims[0][1];

            Rect crop_rect = (Rect) {
                .x0 = bbox->x_min * in->width,
                .y0 = bbox->y_min * in->height,
                .x1 = bbox->x_max * in->width,
                .y1 = bbox->y_max * in->height,
            };

            // care interested object class only
            if (preproc && preproc->object_class && bbox->label_buf) {
                LabelsArray *array = (LabelsArray *)bbox->label_buf->data;
                if (bbox->label_id >= array->num) {
                    av_log(NULL, AV_LOG_ERROR, "The json file must match the input model\n");
                    ret = AVERROR(ERANGE);
                    goto fail;
                }
                if (0 != strcmp(preproc->object_class, array->label[bbox->label_id]))
                    continue;
            }

            if (vpp->device == VPP_DEVICE_SW) {
                ret = vpp->sw_vpp->crop_and_scale(in, &crop_rect,
                        scale_width, scale_height,
                        vpp->expect_format, tmp->data, tmp->linesize);
            } else {
#if CONFIG_VAAPI
                ret = vpp->va_vpp->crop_and_scale(vpp->va_vpp, in, &crop_rect,
                        scale_width, scale_height, tmp->data, tmp->linesize);
#endif
            }
            if (ret != 0) {
                ret = AVERROR(EINVAL);
                goto fail;
            }

            // TODO: support dynamic batch for faces
            ff_inference_base_submit_frame(base, tmp, 0, 0);
            ff_inference_base_infer(base);

            for (output = 0; output < oinfo->number; output++) {
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
    s->frame_number++;
    return ff_filter_frame(outlink, in);
fail:
    if (c_array)
        av_freep(&c_array);
    if (c_meta)
        av_freep(&c_meta);
    av_frame_free(&in);
    return ret;
}

static av_cold int config_input(AVFilterLink *inlink)
{
    int i, ret;
    AVFrame *frame;

    AVFilterContext             *ctx = inlink->dst;
    InferenceClassifyContext      *s = ctx->priv;
    enum AVPixelFormat expect_format = AV_PIX_FMT_BGR24;
    const AVPixFmtDescriptor   *desc = av_pix_fmt_desc_get(inlink->format);

    if (!desc)
        return AVERROR(EINVAL);

    for (i = 0; i < s->loaded_num; i++) {
        InferenceBaseContext *base = s->infer_bases[i];
        DNNModelInfo         *info = ff_inference_base_get_input_info(base);
        VideoPP               *vpp = ff_inference_base_get_vpp(base);

        int input_width  = info->dims[0][0];
        int input_height = info->dims[0][1];

        // right now, no model needs multiple inputs
        // av_assert0(info->number == 1);

        ff_inference_dump_model_info(ctx, info);

        vpp->device = (desc->flags & AV_PIX_FMT_FLAG_HWACCEL) ?
            VPP_DEVICE_HW : VPP_DEVICE_SW;

        // allocate avframes to save preprocessed data
        frame = av_frame_alloc();
        if (!frame)
            return AVERROR(ENOMEM);
        frame->width   = input_width;
        frame->height  = input_height;
        frame->format  = expect_format;
        vpp->frames[0] = frame;

        if (vpp->device == VPP_DEVICE_SW) {
            ret = av_frame_get_buffer(frame, 0);
            if (ret < 0)
                goto fail;
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

            ret = va_vpp_surface_alloc(vpp->va_vpp,
                    input_width, input_height, s->vpp_format);
            if (ret < 0) {
                av_log(ctx, AV_LOG_ERROR, "Create va surface failed\n");
                ret = AVERROR(EINVAL);
                goto fail;
            }

            frame->format = vpp->va_vpp->av_format;
#endif
        }
    }

    return 0;
fail:
    for (i = 0; i < s->loaded_num; i++) {
        VideoPP *vpp = ff_inference_base_get_vpp(s->infer_bases[i]);

        frame = vpp->frames[0];
        if (!frame)
            continue;

        av_frame_free(&frame);

#if CONFIG_VAAPI
        if (vpp->va_vpp) {
            va_vpp_device_free(vpp->va_vpp);
            av_freep(&vpp->va_vpp);
        }
#endif
    }
    return ret;
}

static av_cold int config_output(AVFilterLink *outlink)
{
    int i;
    AVFilterContext        *ctx = outlink->src;
    InferenceClassifyContext *s = ctx->priv;

    for (i = 0; i < s->loaded_num; i++) {
        InferenceBaseContext *base = s->infer_bases[i];
        DNNModelInfo *info = ff_inference_base_get_output_info(base);
        ff_inference_dump_model_info(ctx, info);

#if CONFIG_VAAPI
        if (!outlink->hw_frames_ctx) {
            VideoPP *vpp = ff_inference_base_get_vpp(base);
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
        }
#endif
    }

    return 0;
}

static const AVOption inference_classify_options[] = {
    { "dnn_backend",    "DNN backend for model execution", OFFSET(backend_type),    AV_OPT_TYPE_FLAGS,  { .i64 = DNN_INTEL_IE },          0, 2,    FLAGS, "engine" },
    { "model",          "path to model files for network", OFFSET(model_file),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "model_proc",     "model preproc and postproc",      OFFSET(model_proc),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "vpp_format",     "specify vpp output format",       OFFSET(vpp_format),      AV_OPT_TYPE_STRING, { .str = NULL},                   0, 0,    FLAGS },
    { "device",         "running on device type",          OFFSET(device_type),     AV_OPT_TYPE_FLAGS,  { .i64 = DNN_TARGET_DEVICE_CPU }, 0, 12,   FLAGS },
    { "interval",       "do infer every Nth frame",        OFFSET(every_nth_frame), AV_OPT_TYPE_INT,    { .i64 = 1 },                     1, 1024, FLAGS },
    { "batch_size",     "batch size per infer",            OFFSET(batch_size),      AV_OPT_TYPE_INT,    { .i64 = 1 },                     1, 1024, FLAGS },
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
