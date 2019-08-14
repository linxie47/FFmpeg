/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include <libavutil/frame.h>
#include "libavfilter/avfilter.h"
#include <json-c/json.h>

#define OFFSET(x) offsetof(MetaConvertContext, x)
#define FLAGS (AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_FILTERING_PARAM)

typedef struct MetaConvertContext {
    const AVClass *class;

    char *model;
    char *converter;
    char *method;
    char *location;
    char *layer;

    int frame_number;
    FILE *f;

    int (*convert_func)(AVFilterContext *ctx, AVFrame *frame, json_object *info_object);

} MetaConvertContext;

int detection_to_json(AVFilterContext *ctx, AVFrame *frame, json_object *info_object);

int classification_to_json(AVFilterContext *ctx, AVFrame *frame, json_object *info_object);

int all_to_json(AVFilterContext *ctx, AVFrame *frame, json_object *info_object);

int tensors_to_file(AVFilterContext *ctx, AVFrame *frame, json_object *info_object);

int convert_roi_detection(json_object *info_object, AVFrame *frame);

int convert_roi_tensor(json_object *info_object, AVFrame *frame);