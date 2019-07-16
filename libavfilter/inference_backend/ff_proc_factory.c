/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "ff_proc_factory.h"
#include <libavutil/avassert.h>

static void infer_detect_metadata_buffer_free(void *opaque, uint8_t *data) {
    BBoxesArray *bboxes = ((InferDetectionMeta *)data)->bboxes;

    if (bboxes) {
        int i;
        for (i = 0; i < bboxes->num; i++) {
            InferDetection *p = bboxes->bbox[i];
            if (p->label_buf)
                av_buffer_unref(&p->label_buf);
            av_freep(&p);
        }
        av_free(bboxes->bbox);
        av_freep(&bboxes);
    }

    av_free(data);
}

static void infer_classify_metadata_buffer_free(void *opaque, uint8_t *data) {
    int i;
    InferClassificationMeta *meta = (InferClassificationMeta *)data;
    ClassifyArray *classes = meta->c_array;

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

static int get_unbatched_size_in_bytes(OutputBlobContext *blob_ctx, size_t batch_size) {
    const OutputBlobMethod *blob = blob_ctx->output_blob_method;
    size_t size;
    Dimensions *dim = blob->GetDims(blob_ctx);

    if (dim->dims[0] != batch_size) {
        av_log(NULL, AV_LOG_ERROR, "Blob last dimension should be equal to batch size");
        av_assert0(0);
    }
    size = dim->dims[1];
    for (int i = 2; i < dim->num_dims; i++) {
        size *= dim->dims[i];
    }
    switch (blob->GetPrecision(blob_ctx)) {
    case II_FP32:
        size *= sizeof(float);
        break;
    case II_U8:
    default:
        break;
    }
    return size;
}

static void ExtractBoundingBoxes(const OutputBlobArray *blob_array, InferenceROIArray *infer_roi_array,
                                 ModelOutputPostproc *model_postproc, const char *model_name,
                                 const FFBaseInference *ff_base_inference) {
    for (int n = 0; n < blob_array->num_blobs; n++) {
        AVBufferRef *labels = NULL;
        BBoxesArray **boxes = NULL;
        OutputBlobContext *ctx = blob_array->output_blobs[n];
        const OutputBlobMethod *blob = ctx->output_blob_method;

        const char *layer_name = blob->GetOutputLayerName(ctx);
        const float *detections = (const float *)blob->GetData(ctx);

        Dimensions *dim = blob->GetDims(ctx);
        IILayout layout = blob->GetLayout(ctx);

        int object_size = 0;
        int max_proposal_count = 0;
        float threshold = ff_base_inference->param.threshold;

        switch (layout) {
        case II_LAYOUT_NCHW:
            object_size = dim->dims[3];
            max_proposal_count = dim->dims[2];
            break;
        default:
            av_log(NULL, AV_LOG_ERROR, "Unsupported output layout, boxes won't be extracted\n");
            continue;
        }

        if (object_size != 7) { // SSD DetectionOutput format
            av_log(NULL, AV_LOG_ERROR, "Unsupported output dimensions, boxes won't be extracted\n");
            continue;
        }

        if (ff_base_inference->param.model_postproc) {
            int idx = findModelPostProcByName(ff_base_inference->param.model_postproc, layer_name);
            if (idx != MAX_MODEL_OUTPUT)
                labels = ff_base_inference->param.model_postproc->procs[idx].labels;
        }

        boxes = (BBoxesArray **)av_mallocz_array(infer_roi_array->num_infer_ROIs, sizeof(boxes[0]));
        av_assert0(boxes);

        for (int i = 0; i < max_proposal_count; i++) {
            int image_id = (int)detections[i * object_size + 0];
            int label_id = (int)detections[i * object_size + 1];
            float confidence = detections[i * object_size + 2];
            float x_min = detections[i * object_size + 3];
            float y_min = detections[i * object_size + 4];
            float x_max = detections[i * object_size + 5];
            float y_max = detections[i * object_size + 6];
            if (image_id < 0 || (size_t)image_id >= infer_roi_array->num_infer_ROIs)
                break;

            if (confidence < threshold)
                continue;

            if (boxes[image_id] == NULL) {
                boxes[image_id] = (BBoxesArray *)av_mallocz(sizeof(*boxes[image_id]));
                av_assert0(boxes[image_id]);
            }

            /* using integer to represent */
            {
                FFVideoRegionOfInterestMeta *roi = &infer_roi_array->infer_ROIs[image_id]->roi;
                InferDetection *new_bbox = (InferDetection *)av_mallocz(sizeof(*new_bbox));

                int width = roi->w;
                int height = roi->h;
                int ix_min = (int)(x_min * width + 0.5);
                int iy_min = (int)(y_min * height + 0.5);
                int ix_max = (int)(x_max * width + 0.5);
                int iy_max = (int)(y_max * height + 0.5);

                if (ix_min < 0)
                    ix_min = 0;
                if (iy_min < 0)
                    iy_min = 0;
                if (ix_max > width)
                    ix_max = width;
                if (iy_max > height)
                    iy_max = height;

                av_assert0(new_bbox);
                new_bbox->x_min = ix_min;
                new_bbox->y_min = iy_min;
                new_bbox->x_max = ix_max;
                new_bbox->y_max = iy_max;
                new_bbox->confidence = confidence;
                new_bbox->label_id = label_id;
                if (labels)
                    new_bbox->label_buf = av_buffer_ref(labels);
                av_dynarray_add(&boxes[image_id]->bbox, &boxes[image_id]->num, new_bbox);
            }
        }

        for (int n = 0; n < infer_roi_array->num_infer_ROIs; n++) {
            AVBufferRef *ref;
            AVFrame *av_frame;
            AVFrameSideData *sd;

            InferDetectionMeta *detect_meta = (InferDetectionMeta *)av_malloc(sizeof(*detect_meta));
            av_assert0(detect_meta);

            detect_meta->bboxes = boxes[n];

            ref = av_buffer_create((uint8_t *)detect_meta, sizeof(*detect_meta), &infer_detect_metadata_buffer_free,
                                   NULL, 0);
            if (ref == NULL) {
                infer_detect_metadata_buffer_free(NULL, (uint8_t *)detect_meta);
                av_assert0(ref);
            }

            av_frame = infer_roi_array->infer_ROIs[n]->frame;
            // add meta data to side data
            sd = av_frame_new_side_data_from_buf(av_frame, AV_FRAME_DATA_INFERENCE_DETECTION, ref);
            if (sd == NULL) {
                av_buffer_unref(&ref);
                av_assert0(sd);
            }
            av_log(NULL, AV_LOG_DEBUG, "av_frame:%p sd:%d\n", av_frame, av_frame->nb_side_data);
        }

        av_free(boxes);
    }
}

static void ExtractYOLOV3BoundingBoxes(const OutputBlobArray *blob_array, InferenceROIArray *infer_roi_array,
                                       ModelOutputPostproc *model_postproc, const char *model_name,
                                       const FFBaseInference *ff_base_inference) {
}

static int CreateNewClassifySideData(AVFrame *frame, InferClassificationMeta *classify_meta) {
    AVBufferRef *ref;
    AVFrameSideData *new_sd;
    ref = av_buffer_create((uint8_t *)classify_meta, sizeof(*classify_meta), &infer_classify_metadata_buffer_free, NULL,
                           0);
    if (!ref)
        return AVERROR(ENOMEM);

    // add meta data to side data
    new_sd = av_frame_new_side_data_from_buf(frame, AV_FRAME_DATA_INFERENCE_CLASSIFICATION, ref);
    if (!new_sd) {
        av_buffer_unref(&ref);
        av_log(NULL, AV_LOG_ERROR, "Could not add new side data\n");
        return AVERROR(ENOMEM);
    }

    return 0;
}

static av_cold void dump_softmax(char *name, int label_id, float conf, AVBufferRef *label_buf) {
    LabelsArray *array = (LabelsArray *)label_buf->data;

    av_log(NULL, AV_LOG_DEBUG, "CLASSIFY META - Label id:%d %s:%s Conf:%f\n", label_id, name, array->label[label_id],
           conf);
}

static av_cold void dump_tensor_value(char *name, float value) {
    av_log(NULL, AV_LOG_DEBUG, "CLASSIFY META - %s:%1.2f\n", name, value);
}

static void find_max_element_index(const float *array, int len, int *index, float *value) {
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

static int attributes_to_text(FFVideoRegionOfInterestMeta *meta, OutputPostproc *post_proc, void *data, Dimensions *dim,
                              InferClassification *classification, InferClassificationMeta *classify_meta) {
    const float *blob_data = (const float *)data;
    uint32_t method_max, method_compound, method_index;

    method_max = !strcmp(post_proc->method, "max");
    method_compound = !strcmp(post_proc->method, "compound");
    method_index = !strcmp(post_proc->method, "index");

    if (!blob_data)
        return -1;

    if (method_max) {
        int index;
        float confidence;
        size_t n = dim->dims[1];

        find_max_element_index(data, n, &index, &confidence);

        classification->detect_id = meta->index;
        classification->name = post_proc->attribute_name;
        classification->label_id = index;
        classification->confidence = confidence;
        classification->label_buf = av_buffer_ref(post_proc->labels);

        if (classification->label_buf) {
            dump_softmax(classification->name, classification->label_id, classification->confidence,
                         classification->label_buf);
        }
    } else if (method_compound) {
        int i;
        double threshold = 0.5;
        float confidence = 0;
        char attributes[4096] = {};
        LabelsArray *array;

        if (post_proc->threshold != 0)
            threshold = post_proc->threshold;

        array = (LabelsArray *)post_proc->labels->data;
        for (i = 0; i < array->num; i++) {
            if (blob_data[i] >= threshold)
                strncat(attributes, array->label[i], (strlen(array->label[i]) + 1));
            if (blob_data[i] > confidence)
                confidence = blob_data[i];
        }

        classification->name = post_proc->attribute_name;
        classification->confidence = confidence;

        av_log(NULL, AV_LOG_DEBUG, "Attributes: %s\n", attributes);
    } else if (method_index) {
        int i;
        char attributes[1024] = {};
        LabelsArray *array;

        array = (LabelsArray *)post_proc->labels->data;
        for (i = 0; i < array->num; i++) {
            int value = blob_data[i];
            if (value < 0 || value >= array->num)
                break;
            strncat(attributes, array->label[value], (strlen(array->label[value]) + 1));
        }

        classification->name = post_proc->attribute_name;

        av_log(NULL, AV_LOG_DEBUG, "Attributes: %s\n", attributes);
    }

    return 0;
}

static int tensor_to_text(FFVideoRegionOfInterestMeta *meta, OutputPostproc *post_proc, void *data, Dimensions *dim,
                          InferClassification *classification, InferClassificationMeta *classify_meta) {
    // InferClassification *classify;
    const float *blob_data = (const float *)data;
    double scale = 1.0;

    if (!blob_data)
        return -1;

    if (post_proc->tensor2text_scale != 0)
        scale = post_proc->tensor2text_scale;

    classification->detect_id = meta->index;
    classification->name = post_proc->attribute_name;
    classification->value = *blob_data * scale;

    dump_tensor_value(classification->name, classification->value);
    return 0;
}

static void Blob2RoiMeta(const OutputBlobArray *blob_array, InferenceROIArray *infer_roi_array,
                         ModelOutputPostproc *model_postproc, const char *model_name,
                         const FFBaseInference *ff_base_inference) {
    int batch_size = infer_roi_array->num_infer_ROIs;

    for (int n = 0; n < blob_array->num_blobs; n++) {
        OutputBlobContext *ctx = blob_array->output_blobs[n];
        const OutputBlobMethod *blob;
        const char *layer_name;
        uint8_t *data = NULL;
        int size;
        OutputPostproc *post_proc = NULL;
        Dimensions *dimensions = NULL;

        av_assert0(ctx);

        blob = ctx->output_blob_method;
        layer_name = blob->GetOutputLayerName(ctx);
        data = (uint8_t *)blob->GetData(ctx);
        dimensions = blob->GetDims(ctx);
        size = get_unbatched_size_in_bytes(ctx, batch_size);

        if (model_postproc) {
            int proc_idx = findModelPostProcByName(model_postproc, layer_name);
            if (proc_idx != MAX_MODEL_OUTPUT)
                post_proc = &model_postproc->procs[proc_idx];
        }

        for (int b = 0; b < batch_size; b++) {
            FFVideoRegionOfInterestMeta *meta = &infer_roi_array->infer_ROIs[b]->roi;
            AVFrame *av_frame = infer_roi_array->infer_ROIs[b]->frame;
            AVFrameSideData *sd = NULL;
            InferClassificationMeta *classify_meta = NULL;
            InferClassification *classification = NULL;

            sd = av_frame_get_side_data(av_frame, AV_FRAME_DATA_INFERENCE_CLASSIFICATION);
            if (sd) {
                // append to exsiting side data
                classify_meta = (InferClassificationMeta *)sd->data;
                av_assert0(classify_meta);
            } else {
                ClassifyArray *classify_array = NULL;
                // new classification meta data
                classify_meta = av_mallocz(sizeof(*classify_meta));
                classify_array = av_mallocz(sizeof(*classify_array));
                av_assert0(classify_meta && classify_array);
                classify_meta->c_array = classify_array;
                av_assert0(0 == CreateNewClassifySideData(av_frame, classify_meta));
            }

            classification = av_mallocz(sizeof(*classification));
            av_assert0(classification);
            classification->layer_name = (char *)layer_name;
            classification->model = (char *)model_name;

            if (post_proc && post_proc->converter) {
                if (!strcmp(post_proc->converter, "attributes")) {
                    attributes_to_text(meta, post_proc, (void *)(data + b * size), dimensions, classification,
                                       classify_meta);
                } else if (!strcmp(post_proc->converter, "tensor2text")) {
                    tensor_to_text(meta, post_proc, (void *)(data + b * size), dimensions, classification,
                                   classify_meta);
                } else {
                    av_log(NULL, AV_LOG_ERROR, "Undefined converter:%s\n", post_proc->converter);
                    break;
                }
            } else {
                // copy data to tensor buffer
                classification->detect_id = meta->index;
                classification->name = (char *)"default";
                classification->tensor_buf = av_buffer_alloc(size);
                av_assert0(classification->tensor_buf);
                memcpy(classification->tensor_buf->data, data + b * size, size);
            }

            av_dynarray_add(&classify_meta->c_array->classifications, &classify_meta->c_array->num, classification);
        }
    }
}

PostProcFunction getPostProcFunctionByName(const char *name, const char *model) {
    if (name == NULL || model == NULL)
        return NULL;

    if (!strcmp(name, "ie_detect")) {
        if (strstr(model, "yolo"))
            return (PostProcFunction)ExtractYOLOV3BoundingBoxes;
        else
            return (PostProcFunction)ExtractBoundingBoxes;
    } else if (!strcmp(name, "ie_classify")) {
        return (PostProcFunction)Blob2RoiMeta;
    }
    return NULL;
}

int findModelPostProcByName(ModelOutputPostproc *model_postproc, const char *layer_name) {
    int proc_id;
    // search model postproc
    for (proc_id = 0; proc_id < MAX_MODEL_OUTPUT; proc_id++) {
        char *proc_layer_name = model_postproc->procs[proc_id].layer_name;
        // skip this output process
        if (!proc_layer_name)
            continue;
        if (!strcmp(layer_name, proc_layer_name))
            return proc_id;
    }

    av_log(NULL, AV_LOG_DEBUG, "Could not find proc:%s\n", layer_name);
    return proc_id;
}