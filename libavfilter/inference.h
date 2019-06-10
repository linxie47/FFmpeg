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

#if CONFIG_VAAPI
#include <va/va.h>
#endif

#include "libavutil/common.h"
#include "libswscale/swscale.h"
#include "libavutil/hwcontext.h"
#if CONFIG_VAAPI
#include "libavutil/hwcontext_vaapi.h"
#endif

#include "dnn_interface.h"

typedef struct _InferenceBaseContext InferenceBaseContext;
typedef struct _InputPreproc         ModelInputPreproc;
typedef struct _OutputPostproc       OutputPostproc;
typedef struct _ModelOutputPostproc  ModelOutputPostproc;

typedef int (*InferencePreProcess)(InferenceBaseContext *base, int index, AVFrame *in, AVFrame **out);

#define UNUSED(x) (void)(x)

typedef struct InferenceParam {
    char  *model_file;
    char  *labels_file;
    int    backend_type;
    int    device_type;
    char  *cpu_extension;
    char  *gpu_extension;

    int    batch_size;

    // TODO: inputs attributes are different
    int    input_layout;
    int    input_precision;
    int    input_is_image; //!< image or data

    InferencePreProcess preprocess;
} InferenceParam;

#define MAX_VPP_NUM DNN_INPUT_OUTPUT_NUM

/*
 * Vpp device type detected according to frame format
 */
typedef enum { VPP_DEVICE_HW, VPP_DEVICE_SW } VPPDevice;

typedef struct _SwVpp    SwVpp;

typedef struct _VAAPIVpp VAAPIVpp;

/*
 * Generic rectangle structure consists of two diagonal points
 */
typedef struct Rect {
    float x0; float y0; float x1; float y1;
} Rect;

#if CONFIG_VAAPI
struct _VAAPIVpp {
    AVVAAPIDeviceContext *hwctx;
    AVBufferRef  *hw_frames_ref;
    VASurfaceID   va_surface;
    VAConfigID    va_config;
    VAContextID   va_context;

    VAImageFormat *format_list; //!< Surface formats which can be used with this device.
    int            nb_formats;

    VAImage            va_image;
    VAImageFormat      va_format_selected;
    enum AVPixelFormat av_format;

    int  (*scale)(VAAPIVpp *va_vpp, AVFrame *input,
                  int scale_w,      int scale_h,
                  uint8_t *data[],  int stride[]);

    int  (*crop_and_scale)(VAAPIVpp *va_vpp, AVFrame *input,
                           Rect *crop_rect,
                           int scale_w, int scale_h,
                           uint8_t *data[],  int stride[]);
};
#endif

struct _SwVpp {
    struct SwsContext *scale_contexts[MAX_VPP_NUM];

    int  (*scale)(struct SwsContext *context,
                  const uint8_t * const srcSlice[],
                  const int srcStride[], int srcSliceY,
                  int srcSliceH, uint8_t *const dst[],
                  const int dstStride[]);

    int  (*crop_and_scale)(AVFrame *frame, Rect *crop_rect,
                           int   scale_w,  int   scale_h,
                           enum AVPixelFormat scale_format,
                           uint8_t *dst[], int   dstStride[]);
};

typedef struct VideoPP {
    int       device;
    int       expect_format;
    AVFrame  *frames[MAX_VPP_NUM];  ///<! frames to save vpp output
    SwVpp    *sw_vpp;
#if CONFIG_VAAPI
    VAAPIVpp *va_vpp;
#endif
} VideoPP;

struct _InputPreproc {
    int   color_format;     ///<! input data format
    char *layer_name;       ///<! layer name of input
    char *object_class;     ///<! interested object class
};

struct _OutputPostproc {
    char  *layer_name;
    char  *converter;
    char  *attribute_name;
    char  *method;
    double threshold;
    double tensor2text_scale;
    int    tensor2text_precision;
    AVBufferRef *labels;
};

#define MAX_MODEL_OUTPUT 4
struct _ModelOutputPostproc {
    OutputPostproc procs[MAX_MODEL_OUTPUT];
};

#define MAX_TENSOR_DIM_NUM 4
typedef struct InferTensorMeta {
    size_t  dim_size;
    size_t  dims[MAX_TENSOR_DIM_NUM];
    int     layout;
    int     precision;
    char   *layer_name;
    char   *model_name;
    void   *data;
    size_t  total_bytes;
    // AVBufferRef *labels;
} InferTensorMeta;

typedef struct InferDetection {
    float   x_min;
    float   y_min;
    float   x_max;
    float   y_max;
    float   confidence;
    int     label_id;
    int     object_id;
    AVBufferRef *label_buf;
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
    BBoxesArray *bboxes;
} InferDetectionMeta;

typedef struct InferClassification {
    int     detect_id;        ///< detected bbox index
    char   *name;             ///< class name, e.g. emotion, age
    char   *layer_name;       ///< output layer name
    char   *model;            ///< model name
    int     label_id;         ///< label index in labels
    float   confidence;
    float   value;
    AVBufferRef *label_buf;   ///< label buffer
    AVBufferRef *tensor_buf;  ///< output tensor buffer
} InferClassification;

/* dynamic classifications array */
typedef struct ClassifyArray {
    InferClassification **classifications;
    int                   num;
} ClassifyArray;

typedef struct InferClassificationMeta {
    ClassifyArray *c_array;
} InferClassificationMeta;

/* split strings by delimiter */
void av_split(char *str, const char *delim, char **array, int *num, int max);

/* 2-dimensional norm */
double av_norm(float vec[], size_t num);

/* Dot Product */
double av_dot(float vec1[], float vec2[], size_t num);

int ff_inference_base_create(AVFilterContext *avctx, InferenceBaseContext **base, InferenceParam *p);

int ff_inference_base_free(InferenceBaseContext **base);

int ff_inference_base_submit_frame(InferenceBaseContext *base, AVFrame *frame, int input_idx, int batch_idx);

int ff_inference_base_infer(InferenceBaseContext *base);

int ff_inference_base_filter_frame(InferenceBaseContext *base, AVFrame *in);

int ff_inference_base_get_infer_result(InferenceBaseContext *base, int index, InferTensorMeta *metadata);

DNNModelInfo* ff_inference_base_get_input_info(InferenceBaseContext *base);
DNNModelInfo* ff_inference_base_get_output_info(InferenceBaseContext *base);
VideoPP*      ff_inference_base_get_vpp(InferenceBaseContext *base);

void ff_inference_dump_model_info(void *ctx, DNNModelInfo *info);

#if CONFIG_VAAPI
int va_vpp_device_create(VAAPIVpp *ctx, AVFilterLink *inlink);

int va_vpp_device_free(VAAPIVpp *ctx);

int va_vpp_surface_alloc(VAAPIVpp *ctx, size_t width, size_t height, const char *format);

int va_vpp_surface_release(VAAPIVpp *ctx);
#endif

int ff_get_file_size(FILE *fp);

#if CONFIG_LIBJSON_C
void *ff_read_model_proc(const char *path);

void ff_load_default_model_proc(ModelInputPreproc *preproc, ModelOutputPostproc *postproc);

int ff_parse_input_preproc(const void *json, ModelInputPreproc *m_preproc);

int ff_parse_output_postproc(const void *json, ModelOutputPostproc *m_postproc);

void ff_release_model_proc(const void *json, ModelInputPreproc *preproc, ModelOutputPostproc *postproc);
#endif

#endif
