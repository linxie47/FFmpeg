/*
 * Copyright (c) 2018 Sergey Lavrushkin
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
 * DNN inference engine interface.
 */

#ifndef AVFILTER_DNN_INTERFACE_H
#define AVFILTER_DNN_INTERFACE_H

#include "dnn_data.h"

typedef enum {DNN_SUCCESS, DNN_ERROR} DNNReturnType;

typedef enum {DNN_NATIVE, DNN_TF, DNN_INTEL_IE} DNNBackendType;

typedef struct DNNData{
    float *data;
    int width, height, channels;
} DNNData;

typedef struct DNNModel{
    // Stores model that can be different for different backends.
    void *model;
    // Sets model input and output, while allocating additional memory for intermediate calculations.
    // Should be called at least once before model execution.
    DNNReturnType (*set_input_output)(void *model, DNNData *input, DNNData *output);

    // Get the result after the model execuation. Returns DNN_ERROR otherwise. the result is stored in the result->data. the backend is responsible to fill the other structure fields.
    // The user should parse the result independently according to the output data structure format. the structure are defined by the user.
    DNNReturnType (*get_execute_result)(void *model, DNNIOData *result);
    // Set/feed the model with specified input data. Returns DNN_ERROR otherwise.
    DNNReturnType (*set_input)(void *model, const DNNIOData *input);
    // Get the input info of the model. Returns DNN_ERROR otherwise.
    DNNReturnType (*get_input_info)(void *model, DNNModelInfo *info);
    // Set the input info of the model. Returns DNN_ERROR otherwise.
    DNNReturnType (*set_input_info)(void *model, DNNModelInfo *info);
    // Get the output info of the model. Returns DNN_ERROR otherwise.
    DNNReturnType (*get_output_info)(void *model, DNNModelInfo *info);
    // Set the output info of the model. Returns DNN_ERROR otherwise.
    DNNReturnType (*set_output_info)(void *model, DNNModelInfo *info);
    // the model/NN will be created layer by layer according to the model backend type and model graph
    DNNReturnType (*create_model)(void *model);
} DNNModel;

// Stores pointers to functions for loading, executing, freeing DNN models for one of the backends.
typedef struct DNNModule{
    // Loads model and parameters from given file. Returns NULL if it is not possible.
    DNNModel *(*load_model)(const char *model_filename);
    // Executes model with specified input and output. Returns DNN_ERROR otherwise.
    DNNReturnType (*execute_model)(const DNNModel *model);
    // Frees memory allocated for model.
    void (*free_model)(DNNModel **model);

    // Loads model and parameters from given configuration. Returns NULL if it is not possible.
    DNNModel *(*load_model_with_config)(void *config);
} DNNModule;

// Initializes DNNModule depending on chosen backend.
DNNModule *ff_get_dnn_module(DNNBackendType backend_type);

#endif
