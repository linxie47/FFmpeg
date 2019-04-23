/*
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

#ifndef AVFILTER_DNN_DATA_H
#define AVFILTER_DNN_DATA_H

#include <stdint.h>
#include <stddef.h>

/**
* @enum TargetDevice
* @brief Describes known device types
*/
typedef enum DNNTargetDeviceType {
    DNN_TARGET_DEVICE_DEFAULT = 0,
    DNN_TARGET_DEVICE_BALANCED = 1,
    DNN_TARGET_DEVICE_CPU = 2,
    DNN_TARGET_DEVICE_GPU = 3,
    DNN_TARGET_DEVICE_FPGA = 4,
    DNN_TARGET_DEVICE_MYRIAD = 5,
    DNN_TARGET_DEVICE_HDDL = 6,
    DNN_TARGET_DEVICE_GNA  = 7,
    DNN_TARGET_DEVICE_HETERO = 8,
} DNNTargetDeviceType;

/**
* @enum Precision
* @brief Describes Precision types
*/
typedef enum DNNDataPrecisionType {
    DNN_DATA_PRECISION_UNSPECIFDNND = 255, /**< Unspecified value. Used by default */
    DNN_DATA_PRECISION_MIXED = 0,  /**< Mixed value. Can be received from network. No applicable for tensors */
    DNN_DATA_PRECISION_FP32 = 10,  /**< 32bit floating point value */
    DNN_DATA_PRECISION_FP16 = 11,  /**< 16bit floating point value */
    DNN_DATA_PRECISION_Q78 = 20,   /**< 16bit specific signed fixed point precision */
    DNN_DATA_PRECISION_I16 = 30,   /**< 16bit signed integer value */
    DNN_DATA_PRECISION_U8 = 40,    /**< 8bit unsigned integer value */
    DNN_DATA_PRECISION_I8 = 50,    /**< 8bit signed integer value */
    DNN_DATA_PRECISION_U16 = 60,   /**< 16bit unsigned integer value */
    DNN_DATA_PRECISION_I32 = 70,   /**< 32bit signed integer value */
    DNN_DATA_PRECISION_CUSTOM = 80 /**< custom precision has it's own name and size of elements */
} DNNDataPrecisionType;

/**
* @enum Layout
* @brief Layouts that the inference engine supports
*/
typedef enum DNNDataLayoutType {
    DNN_DATA_LAYOUT_ANY = 0,// "any" layout
    DNN_DATA_LAYOUT_NCHW = 1,// I/O data layouts
    DNN_DATA_LAYOUT_NHWC = 2,
    DNN_DATA_LAYOUT_OIHW = 64,// weight layouts
    DNN_DATA_LAYOUT_C = 96,// bias layouts
    DNN_DATA_LAYOUT_CHW = 128,// Single image layout (for mean image)
    DNN_DATA_LAYOUT_HW = 192, // 2D
    DNN_DATA_LAYOUT_NC = 193,
    DNN_DATA_LAYOUT_CN = 194,
    DNN_DATA_LAYOUT_BLOCKED = 200,
    DNN_DATA_LAYOUT_1D = 201, //1D output only
} DNNDataLayoutType;

/**
* @enum Memory Type
* @brief memory type that the inference engine supports?
*/
typedef enum DNNMemoryType {
    DNN_MEM_DEFAULT = 0,
    DNN_MEM_HOST = 1,
    DNN_MEM_GPU = 2,
    DNN_MEM_SHARED = 3,
    DNN_MEM_OTHERS = 4,
} DNNMemoryType;

/**
* @enum Model data format
*/
typedef enum DNNDataFormat {
    DNN_DATA_BGR_PACKED,
    DNN_DATA_BGR_PLANAR,
    DNN_DATA_BGRA_PACKED,
    DNN_DATA_BGRA_PLANAR,
    DNN_DATA_RGB_PACKED,
    DNN_DATA_RGB_PLANAR,
    DNN_DATA_GRAY_PLANAR, /* single channel*/
    DNN_DATA_GENERIC_1D,  /* single channel 1D height/height_stride/channels are 1, output only*/
    DNN_DATA_GENERIC_2D,  /* single channel 2D*/
} DNNDataFormat;

/**
* @structure for DNN device
*/
typedef struct DNNDevice {
    DNNTargetDeviceType type;
    const char * name;
} DNNDevice;

/*
* @struct inference engine Data(image etc) for input and output
* @brief input/output data for the inference engine supports, it is design for 1D/2D data
* spencial for single 1D: height/height_stride/channels are 1 width_stride=width, output only
*/
typedef struct DNNIOData {
#define NUM_DATA_POINTS 4
    uint8_t *data[NUM_DATA_POINTS];
    int  linesize[NUM_DATA_POINTS];
    unsigned int size;      // size=width x height x channels,it is for 1D output/input. unit is byte.
    unsigned int width;
    unsigned int height;
    unsigned int channels;
    // the index of the batch when batch size is bigger than 1. default value is zero when batch size is 1.
    unsigned int batch_idx;
    unsigned int is_image;
    // it describe the data belong to the index of input/output for the model. Defatult value is 0.
    unsigned int in_out_idx;
    DNNDataPrecisionType precision; //DNN_DATA_PRECISION_FP32 or FP16 etc
    DNNMemoryType memory_type;
    DNNDataFormat data_format;
} DNNIOData;

/**
* @struct model input info
* @brief model input info
*/
#define DNN_INPUT_OUTPUT_NUM 8
typedef struct DNNModelInfo {
    char  *layer_name[DNN_INPUT_OUTPUT_NUM];
    size_t       dims[DNN_INPUT_OUTPUT_NUM][4];

    DNNDataPrecisionType precision[DNN_INPUT_OUTPUT_NUM];
    DNNDataLayoutType layout[DNN_INPUT_OUTPUT_NUM];

    // 0 non-image; 1 image.
    unsigned int is_image[DNN_INPUT_OUTPUT_NUM];
    unsigned int batch_size;
    unsigned int number;
} DNNModelInfo;

/**
* @struct model Configuration for the backend of Intel Inference engine
* @brief Configuration for the model of Intel Inference engine
*/
typedef struct DNNModelIntelIEConfig {
    char *model;
    char *labels;
    int   device;
    int   batch_size;
    char *cpu_extension;
    char *gpu_extension;
} DNNModelIntelIEConfig;

#endif
