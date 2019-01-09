/*
 * Copyright (c) 2018 Pengfei Qu, Lin Xie
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
 * DNN inference functions interface for intel inference engine backend.
 */

#include "dnn_backend_intel_ie.h"
#include "libavformat/avio.h"
#include <ie_api_wrapper.h>

typedef struct DNNIntelIEModel {
    void *context;
    IEConfig config;
    IEInputOutputInfo *input_infos;
    IEInputOutputInfo *output_infos;
} DNNIntelIEModel;

static IETargetDeviceType get_device_type_id(DNNTargetDeviceType device_type) {
    switch (device_type) {
    case DNN_TARGET_DEVICE_DEFAULT:
        return IE_Default;
    case DNN_TARGET_DEVICE_BALANCED:
        return IE_Balanced;
    case DNN_TARGET_DEVICE_CPU:
        return IE_CPU;
    case DNN_TARGET_DEVICE_GPU:
        return IE_GPU;
    case DNN_TARGET_DEVICE_FPGA:
        return IE_FPGA;
    case DNN_TARGET_DEVICE_MYRIAD:
        return IE_MYRIAD;
    case DNN_TARGET_DEVICE_HDDL:
        return IE_HDDL;
    case DNN_TARGET_DEVICE_GNA:
        return IE_GNA;
    case DNN_TARGET_DEVICE_HETERO:
        return IE_HETERO;
    default:
        return IE_Default;
    }
}

static IELayoutType get_layout(DNNDataLayoutType layout)
{
    switch (layout) {
    case DNN_DATA_LAYOUT_NCHW:
        return IE_NCHW;
    case DNN_DATA_LAYOUT_NHWC:
        return IE_NHWC;
    case DNN_DATA_LAYOUT_OIHW:
        return IE_OIHW;
    case DNN_DATA_LAYOUT_C:
        return IE_C;
    case DNN_DATA_LAYOUT_CHW:
        return IE_CHW;
    case DNN_DATA_LAYOUT_HW:
        return IE_HW;
    case DNN_DATA_LAYOUT_NC:
        return IE_NC;
    case DNN_DATA_LAYOUT_CN:
        return IE_CN;
    case DNN_DATA_LAYOUT_BLOCKED:
        return IE_BLOCKED;
    case DNN_DATA_LAYOUT_ANY:
        return IE_ANY;
    case DNN_DATA_LAYOUT_1D:
        return IE_ANY;
    default:
        return IE_ANY;
    }
}

static DNNDataLayoutType get_dnn_layout(IELayoutType layout)
{
    switch (layout) {
    case IE_NCHW:
        return DNN_DATA_LAYOUT_NCHW;
    case IE_NHWC:
        return DNN_DATA_LAYOUT_NHWC;
    case IE_OIHW:
        return DNN_DATA_LAYOUT_OIHW;
    case IE_C:
        return DNN_DATA_LAYOUT_C;
    case IE_CHW:
        return DNN_DATA_LAYOUT_CHW;
    case IE_HW:
        return DNN_DATA_LAYOUT_HW;
    case IE_NC:
        return DNN_DATA_LAYOUT_NC;
    case IE_CN:
        return DNN_DATA_LAYOUT_CN;
    case IE_BLOCKED:
        return DNN_DATA_LAYOUT_BLOCKED;
    case IE_ANY:
        return DNN_DATA_LAYOUT_ANY;
    default:
        return DNN_DATA_LAYOUT_ANY;
    }
}

static IEPrecisionType get_precision(DNNDataPrecisionType precision)
{
    switch (precision) {
    case DNN_DATA_PRECISION_MIXED:
        return IE_MIXED;
    case DNN_DATA_PRECISION_FP32:
        return IE_FP32;
    case DNN_DATA_PRECISION_FP16:
        return IE_FP16;
    case DNN_DATA_PRECISION_Q78:
        return IE_Q78;
    case DNN_DATA_PRECISION_I16:
        return IE_I16;
    case DNN_DATA_PRECISION_U8:
        return IE_U8;
    case DNN_DATA_PRECISION_I8:
        return IE_I8;
    case DNN_DATA_PRECISION_U16:
        return IE_U16;
    case DNN_DATA_PRECISION_I32:
        return IE_I32;
    case DNN_DATA_PRECISION_CUSTOM:
        return IE_CUSTOM;
    case DNN_DATA_PRECISION_UNSPECIFDNND:
        return IE_UNSPECIFIED;
    default:
        return IE_FP32;
    }
}

static DNNDataPrecisionType get_dnn_precision(IEPrecisionType precision)
{
    switch (precision) {
    case IE_MIXED:
        return DNN_DATA_PRECISION_MIXED;
    case IE_FP32:
        return DNN_DATA_PRECISION_FP32;
    case IE_FP16:
        return DNN_DATA_PRECISION_FP16;
    case IE_Q78:
        return DNN_DATA_PRECISION_Q78;
    case IE_I16:
        return DNN_DATA_PRECISION_I16;
    case IE_U8:
        return DNN_DATA_PRECISION_U8;
    case IE_I8:
        return DNN_DATA_PRECISION_I8;
    case IE_U16:
        return DNN_DATA_PRECISION_U16;
    case IE_I32:
        return DNN_DATA_PRECISION_I32;
    case IE_CUSTOM:
        return DNN_DATA_PRECISION_CUSTOM;
    case IE_UNSPECIFIED:
        return DNN_DATA_PRECISION_UNSPECIFDNND;
    default:
        return DNN_DATA_PRECISION_FP32;
    }
}

static IEImageFormatType get_data_format(DNNDataFormat format)
{
    switch (format) {
    case DNN_DATA_BGR_PACKED:
    case DNN_DATA_BGRA_PACKED:
        return IE_IMAGE_BGR_PACKED;
    case DNN_DATA_BGR_PLANAR:
    case DNN_DATA_BGRA_PLANAR:
        return IE_IMAGE_BGR_PLANAR;
    case DNN_DATA_RGB_PACKED:
        return IE_IMAGE_RGB_PACKED;
    case DNN_DATA_RGB_PLANAR:
        return IE_IMAGE_RGB_PLANAR;
    case DNN_DATA_GRAY_PLANAR:
        return IE_IMAGE_GRAY_PLANAR;
    case DNN_DATA_GENERIC_1D:
        return IE_IMAGE_GENERIC_1D;
    case DNN_DATA_GENERIC_2D:
        return IE_IMAGE_GENERIC_2D;
    default:
        return IE_IMAGE_FORMAT_UNKNOWN;
    }
}

static void set_model_config_internal(DNNIntelIEModel *ie_model, DNNModelIntelIEConfig *ie_config)
{
    ie_model->config.targetId      = get_device_type_id(ie_config->device);
    ie_model->config.modelFileName = ie_config->model;
    ie_model->config.cpuExtPath    = ie_config->cpu_extension;
    ie_model->config.cldnnExtPath  = ie_config->gpu_extension;
    ie_model->config.perfCounter   = 0;

    ie_model->input_infos          = &(ie_model->config.inputInfos);
    ie_model->output_infos         = &(ie_model->config.outputInfos);
}

static DNNReturnType get_execute_result_intel_ie(void *model, DNNIOData *result)
{
    unsigned int size = 0;
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model || !result)
        return DNN_ERROR;

    result->data[0] = IEGetResultSpace(ie_model->context, result->in_out_idx, &size);
    if (!result->data)
        return DNN_ERROR;

    result->size = size;
    result->precision = DNN_DATA_PRECISION_FP32;

    return DNN_SUCCESS;
}

static DNNReturnType get_input_info_intel_ie(void *model, DNNModelInfo *info)
{
    int id = 0;
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model || !info)
        return DNN_ERROR;

    IEGetModelInputInfo(ie_model->context, ie_model->input_infos);

    if (ie_model->input_infos->number > DNN_INPUT_OUTPUT_NUM)
        return DNN_ERROR;

    for (id = 0; id < ie_model->input_infos->number; id++) {
        memcpy(&info->dims[id][0],
               &ie_model->input_infos->tensorMeta[id].dims[0],
               4 * sizeof(info->dims[id][0]));

        info->layer_name[id] = ie_model->input_infos->tensorMeta[id].layer_name;
        info->precision[id]  = get_dnn_precision(ie_model->input_infos->tensorMeta[id].precision);
        info->layout[id]     = get_dnn_layout(ie_model->input_infos->tensorMeta[id].layout);
    }
    info->batch_size = ie_model->input_infos->batch_size;
    info->number     = ie_model->input_infos->number;

    return DNN_SUCCESS;
}

static DNNReturnType set_input_info_intel_ie(void *model, DNNModelInfo *info)
{
    int id = 0;
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model || !info || info->number > DNN_INPUT_OUTPUT_NUM)
        return DNN_ERROR;

    // image set to input 0
    ie_model->input_infos->tensorMeta[0].precision = get_precision(info->precision[id]);
    ie_model->input_infos->tensorMeta[0].layout    = get_layout(info->layout[id]);
    ie_model->input_infos->tensorMeta[0].dataType  = info->is_image[id];

    ie_model->input_infos->number = info->number;

    IESetModelInputInfo(ie_model->context, ie_model->input_infos);

    return DNN_SUCCESS;
}

static DNNReturnType get_output_info_intel_ie(void *model, DNNModelInfo *info)
{
    int id = 0;
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model || !info)
        return DNN_ERROR;

    IEGetModelOutputInfo(ie_model->context, ie_model->output_infos);

    if (ie_model->output_infos->number > DNN_INPUT_OUTPUT_NUM)
        return DNN_ERROR;

    for (id = 0; id < ie_model->output_infos->number; id++) {
        memcpy(&info->dims[id][0],
               &ie_model->output_infos->tensorMeta[id].dims[0],
               4 * sizeof(info->dims[id][0]));

        info->layer_name[id] = ie_model->output_infos->tensorMeta[id].layer_name;
        info->precision[id]  = get_dnn_precision(ie_model->output_infos->tensorMeta[id].precision);
        info->layout[id]     = get_dnn_layout(ie_model->output_infos->tensorMeta[id].layout);
    }
    info->batch_size = ie_model->output_infos->batch_size;
    info->number     = ie_model->output_infos->number;

    return DNN_SUCCESS;
}

static DNNReturnType set_input_intel_ie(void *model, const DNNIOData *input)
{
    int i;
    IEData data;
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model || !input)
        return DNN_ERROR;

    memset(&data, 0, sizeof(IEData));

    for (i = 0; i < NUM_DATA_POINTS; i++) {
        data.data[i]     = input->data[i];
        data.linesize[i] = input->linesize[i];
    }
    data.width        = input->width;
    data.height       = input->height;
    data.channelNum   = input->channels;
    data.batchIdx     = input->batch_idx;
    data.precision    = get_precision(input->precision);
    data.memType      = input->memory_type;
    data.dataType     = input->is_image;
    data.imageFormat  = get_data_format(input->data_format);

    IESetInput(ie_model->context, input->in_out_idx, &data);

    return DNN_SUCCESS;
}

static DNNReturnType create_model_intel_ie(void *model)
{
    DNNIntelIEModel *ie_model = (DNNIntelIEModel *)model;

    if (!model)
        return DNN_ERROR;

    IECreateModel(ie_model->context, &ie_model->config);

    return DNN_SUCCESS;
}

DNNModel* ff_dnn_load_model_intel_ie(void *config)
{
    DNNModel *model = NULL;
    DNNIntelIEModel *ie_model = NULL;
    DNNModelIntelIEConfig *ie_config = (DNNModelIntelIEConfig *)config;

    if (!ie_config)
        return NULL;

    model = av_mallocz(sizeof(DNNModel));
    if (!model)
        return NULL;

    ie_model = av_mallocz(sizeof(DNNIntelIEModel));
    if (!ie_model) {
        av_freep(&model);
        return NULL;
    }

    set_model_config_internal(ie_model, ie_config);

    ie_model->context = IEAllocateContext();
    if (!ie_model->context) {
        av_freep(&ie_model);
        av_freep(&model);
        return NULL;
    }

    IELoadModel(ie_model->context, &ie_model->config);

    IESetBatchSize(ie_model->context, ie_config->batch_size);

    model->model              = (void *)ie_model;
    model->get_execute_result = &get_execute_result_intel_ie;
    model->set_input          = &set_input_intel_ie;
    model->get_input_info     = &get_input_info_intel_ie;
    model->set_input_info     = &set_input_info_intel_ie;
    model->get_output_info    = &get_output_info_intel_ie;
    model->create_model       = &create_model_intel_ie;

    return model;
}

DNNReturnType ff_dnn_execute_model_intel_ie(const DNNModel *model)
{
    DNNIntelIEModel *ie_model = NULL;

    if (!model)
        return DNN_ERROR;

    ie_model = (DNNIntelIEModel *)model->model;

    IEForward(ie_model->context, IE_INFER_MODE_SYNC);

    return DNN_SUCCESS;
}

void ff_dnn_free_model_intel_ie(DNNModel** model)
{
    DNNIntelIEModel * ie_model = NULL;

    if (*model) {
        ie_model = (DNNIntelIEModel *)(*model)->model;
        IEFreeContext(ie_model->context);
        av_freep(&ie_model);
        av_freep(model);
    }
}

