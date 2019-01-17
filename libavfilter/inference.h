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

#ifndef AVFILTER_INFERENCE_H
#define AVFILTER_INFERENCE_H

#include "libavutil/common.h"
#include "dnn_interface.h"

typedef struct InferenceBaseContext InferenceBaseContext;

typedef int (*InferencePreProcess)(InferenceBaseContext *base, int index, AVFrame *in, AVFrame **out);

typedef struct InferenceParam {
    char  *model_file;
    char  *labels_file;
    int    backend_type;
    int    device_type;
    char  *cpu_extension;
    char  *gpu_extension;

    int    batch_size;
    int    every_nth_frame;
    float  threshold;

    // TODO: inputs attributes are different
    int    input_layout;
    int    input_precision;
    int    input_is_image; //!< image or data

    InferencePreProcess preprocess;
} InferenceParam;

#define MAX_VPP_NUM DNN_INPUT_OUTPUT_NUM

typedef enum { VPP_DEVICE_HW, VPP_DEVICE_SW } VPPDevice;

typedef struct VideoPP {
    int      device;
    void    *scale_contexts[MAX_VPP_NUM];
    AVFrame *frames[MAX_VPP_NUM];          //<! frames to save vpp output

    int    (*swscale)(void *context,
                      const uint8_t * const srcSlice[],
                      const int srcStride[], int srcSliceY,
                      int srcSliceH, uint8_t *const dst[],
                      const int dstStride[]);
    int    (*crop_and_scale)(AVFrame *frame,
                             float crop_x0,  float crop_y0,
                             float crop_x1,  float crop_y1,
                             int   scale_w,  int   scale_h,
                             uint8_t *dst[], int   dstStride[]);
} VideoPP;

#define MAX_TENSOR_DIM_NUM 8
typedef struct InferTensorMeta {
    size_t  dim_size;
    size_t  dims[MAX_TENSOR_DIM_NUM];
    int     layout;
    int     precision;
    char   *layer_name;
    char   *model_name;
    void   *data;
    size_t  total_bytes;
    AVBufferRef *labels;
} InferTensorMeta;

typedef struct InferDetection {
    float x_min;
    float y_min;
    float x_max;
    float y_max;
    float confidence;
    int   label_id;
    int   object_id;
    AVBufferRef *text;
} InferDetection;

/* dynamic bounding boxes array */
typedef struct BBoxesArray {
    InferDetection **bbox;
    int              num;
} BBoxesArray;

/* dynamic labels array */
typedef struct LabelsArray {
    char **label;
    int    num;
} LabelsArray;

typedef struct InferDetectionMeta {
    LabelsArray *labels;
    BBoxesArray *bboxes;
} InferDetectionMeta;

typedef struct InferClassification {
    int     detect_id;        ///< detected bbox index
    char   *name;             ///< class name, e.g. emotion, age
    int     label_id;         ///< label index in labels
    float   confidence;
    float   value;
    AVBufferRef *label_buf;   ///< label ref buf from label file
} InferClassification;

/* dynamic classifications array */
typedef struct ClassifyArray {
    InferClassification **classifications;
    int                   num;
} ClassifyArray;

typedef struct InferClassificationMeta {
    ClassifyArray *c_array;
} InferClassificationMeta;

int ff_inference_base_create(AVFilterContext *avctx, InferenceBaseContext **base, InferenceParam *p);

int ff_inference_base_free(InferenceBaseContext **base);

int ff_inference_base_submit_frame(InferenceBaseContext *base, AVFrame *frame, int input_idx, int batch_idx);

int ff_inference_base_infer(InferenceBaseContext *base);

int ff_inference_base_filter_frame(InferenceBaseContext *base, AVFrame *in);

int ff_inference_base_get_infer_result(InferenceBaseContext *base, int index, InferTensorMeta *metadata);

DNNModelInfo* ff_inference_base_get_input_info(InferenceBaseContext *base);
DNNModelInfo* ff_inference_base_get_output_info(InferenceBaseContext *base);
VideoPP*      ff_inference_base_get_vpp(InferenceBaseContext *base);

#endif
