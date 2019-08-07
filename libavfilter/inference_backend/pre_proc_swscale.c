/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "pre_proc.h"
#include <assert.h>
#include <libswscale/swscale.h>
// #define DEBUG

#if CONFIG_SWSCALE || HAVE_FFMPEG

#ifdef DEBUG
static void DumpBGRpToFile(const Image *out_image) {
    FILE *fp;
    char file_name[256] = {};
    static int dump_frame_num = 0;

    sprintf(file_name, "ff_pre_proc%03d.rgb", dump_frame_num++);
    fp = fopen(file_name, "w+b");
    assert(fp);

    const uint8_t *b_channel = out_image->planes[0];
    const uint8_t *g_channel = out_image->planes[1];
    const uint8_t *r_channel = out_image->planes[2];

    int size = out_image->height * out_image->width * 3;
    uint8_t *data = (uint8_t *)malloc(size);
    memset(data, 0, size);

    for (int i = 0; i < out_image->height; i++) {
        for (int j = 0; j < out_image->width; j++) {
            data[3 * j + i * 3 * out_image->width] = r_channel[j + i * out_image->width];
            data[3 * j + i * 3 * out_image->width + 1] = g_channel[j + i * out_image->width];
            data[3 * j + i * 3 * out_image->width + 2] = b_channel[j + i * out_image->width];
        }
    }
    fwrite(data, out_image->height * out_image->width * 3, 1, fp);
    free(data);
    fclose(fp);
}

static void DumpRGBpToFile(const Image *out_image) {
    FILE *fp;
    char file_name[256] = {};
    static int dump_frame_num = 0;

    sprintf(file_name, "ff_rgbp_source%03d.rgb", dump_frame_num++);
    fp = fopen(file_name, "w+b");
    assert(fp);

    const uint8_t *b_channel = out_image->planes[2];
    const uint8_t *g_channel = out_image->planes[1];
    const uint8_t *r_channel = out_image->planes[0];

    int size = out_image->height * out_image->width * 3;
    uint8_t *data = (uint8_t *)malloc(size);
    memset(data, 0, size);

    for (int i = 0; i < out_image->height; i++) {
        for (int j = 0; j < out_image->width; j++) {
            data[3 * j + i * 3 * out_image->width] = r_channel[j + i * out_image->width];
            data[3 * j + i * 3 * out_image->width + 1] = g_channel[j + i * out_image->width];
            data[3 * j + i * 3 * out_image->width + 2] = b_channel[j + i * out_image->width];
        }
    }
    fwrite(data, out_image->height * out_image->width * 3, 1, fp);
    free(data);
    fclose(fp);
}

static inline void DumpImageInfo(const Image *p) {
    av_log(NULL, AV_LOG_INFO, "Image w:%d h:%d f:%x, plane: %p %p %p  stride: %d %d %d \n", p->width, p->height,
           p->format, p->planes[0], p->planes[1], p->planes[2], p->stride[0], p->stride[1], p->stride[2]);
}

#endif

static inline enum AVPixelFormat FOURCC2FFmpegFormat(int format) {
    switch (format) {
    case FOURCC_NV12:
        return AV_PIX_FMT_NV12;
    case FOURCC_BGRA:
        return AV_PIX_FMT_BGRA;
    case FOURCC_BGRX:
        return AV_PIX_FMT_BGRA;
    case FOURCC_BGR:
        return AV_PIX_FMT_BGR24;
    case FOURCC_RGBP:
        return AV_PIX_FMT_RGBP;
    case FOURCC_I420:
        return AV_PIX_FMT_YUV420P;
    }
    return AV_PIX_FMT_NONE;
}

typedef struct FFPreProc {
    struct SwsContext *sws_context;
    struct SwsContext *cascaded_context[2];
    Image cascaded_img[2];
} FFPreProc;

static void FFPreProcConvert(PreProcContext *context, const Image *src, Image *dst, int bAllocateDestination) {
    FFPreProc *ff_pre_proc = (FFPreProc *)context->priv;
    struct SwsContext *sws_context = ff_pre_proc->sws_context;
    struct SwsContext **cascaded_context = ff_pre_proc->cascaded_context;
    Image *cascaded_img = ff_pre_proc->cascaded_img;
    uint8_t *gbr_planes[3] = {};
    int i;

    // if identical format and resolution
    if (src->format == dst->format && src->format == FOURCC_RGBP && src->width == dst->width &&
        src->height == dst->height) {
        int planes_count = GetPlanesCount(src->format);
        // RGB->BGR
        Image src_bgr = *src;
        src_bgr.planes[0] = src->planes[2];
        src_bgr.planes[2] = src->planes[0];
        for (int i = 0; i < planes_count; i++) {
            if (src_bgr.width == src_bgr.stride[i]) {
                memcpy(dst->planes[i], src_bgr.planes[i], src_bgr.width * src_bgr.height * sizeof(uint8_t));
            } else {
                int dst_stride = dst->stride[i] * sizeof(uint8_t);
                int src_stride = src_bgr.stride[i] * sizeof(uint8_t);
                for (int r = 0; r < src_bgr.height; r++) {
                    memcpy(dst->planes[i] + r * dst_stride, src_bgr.planes[i] + r * src_stride, dst->width);
                }
            }
        }

        return;
    }
#define PLANE_NUM 3
    if (cascaded_img[0].width != dst->width || cascaded_img[0].height != dst->height) {
        for (i = 0; i < 2; i++) {
            cascaded_img[i].width = dst->width;
            cascaded_img[i].height = dst->height;
            cascaded_img[i].stride[0] = FFALIGN(dst->width * PLANE_NUM, 16);
            if (cascaded_img[i].planes[0])
                free(cascaded_img[i].planes[0]);
            cascaded_img[i].planes[0] = (uint8_t *)malloc(cascaded_img[i].stride[0] * cascaded_img[i].height * PLANE_NUM);
            if (!cascaded_img[i].planes[0]) {
                fprintf(stderr, "Error of mem alloc on FFMPEG sws_scale\n");
                assert(0);
            }
            for (int j = 1; j < PLANE_NUM; j++) {
                cascaded_img[i].stride[j] = cascaded_img[i].stride[0];
                cascaded_img[i].planes[j] = cascaded_img[i].planes[0] + j * (cascaded_img[i].stride[j] * cascaded_img[i].height);
            }
        }
    }

    sws_context = sws_getCachedContext(sws_context, src->width, src->height, FOURCC2FFmpegFormat(src->format),
                                       dst->width, dst->height, AV_PIX_FMT_YUV420P, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    cascaded_context[0] = sws_getCachedContext(cascaded_context[0], dst->width, dst->height, AV_PIX_FMT_YUV420P,
                                       dst->width, dst->height, AV_PIX_FMT_RGB24, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    cascaded_context[1] = sws_getCachedContext(cascaded_context[1], dst->width, dst->height, AV_PIX_FMT_RGB24,
                                       dst->width, dst->height, AV_PIX_FMT_GBRP, SWS_FAST_BILINEAR, NULL, NULL, NULL);
    assert(sws_context);
    assert(cascaded_context[0]);
    assert(cascaded_context[1]);

    // BGR->GBR
    gbr_planes[0] = dst->planes[1];
    gbr_planes[1] = dst->planes[0];
    gbr_planes[2] = dst->planes[2];

    if (!sws_scale(sws_context, (const uint8_t *const *)src->planes, src->stride, 0, src->height, cascaded_img[0].planes,
                   cascaded_img[0].stride)) {
        fprintf(stderr, "Error on FFMPEG sws_scale\n");
        assert(0);
    }
    if (!sws_scale(cascaded_context[0], (const uint8_t *const *)cascaded_img[0].planes, cascaded_img[0].stride, 0, dst->height, cascaded_img[1].planes,
                   cascaded_img[1].stride)) {
        fprintf(stderr, "Error on FFMPEG cascaded_context0\n");
        assert(0);
    }
    if (!sws_scale(cascaded_context[1], (const uint8_t *const *)cascaded_img[1].planes, cascaded_img[1].stride, 0, dst->height, gbr_planes,
                   dst->stride)) {
        fprintf(stderr, "Error on FFMPEG cascaded_context1\n");
        assert(0);
    }

    /* dump pre-processed image to file */
    // DumpBGRpToFile(dst);
    ff_pre_proc->sws_context = sws_context;
    ff_pre_proc->cascaded_context[0] = cascaded_context[0];
    ff_pre_proc->cascaded_context[1] = cascaded_context[1];
}

static void FFPreProcDestroy(PreProcContext *context) {
    FFPreProc *ff_pre_proc = (FFPreProc *)context->priv;
    Image *cascaded_img = ff_pre_proc->cascaded_img;
    int i;

    if (ff_pre_proc->sws_context) {
        sws_freeContext(ff_pre_proc->sws_context);
        ff_pre_proc->sws_context = NULL;
    }
    if (ff_pre_proc->cascaded_context[0]) {
        sws_freeContext(ff_pre_proc->cascaded_context[0]);
        ff_pre_proc->cascaded_context[0] = NULL;
    }
    if (ff_pre_proc->cascaded_context[1]) {
        sws_freeContext(ff_pre_proc->cascaded_context[1]);
        ff_pre_proc->cascaded_context[1] = NULL;
    }
    for (i = 0; i < 2; i++) {
        if (cascaded_img[i].planes[0])
            free(cascaded_img[i].planes[0]);
        cascaded_img[i].planes[0] = NULL;
    }
}

PreProc pre_proc_swscale = {
    .name = "swscale",
    .priv_size = sizeof(FFPreProc),
    .mem_type = MEM_TYPE_SYSTEM,
    .Convert = FFPreProcConvert,
    .Destroy = FFPreProcDestroy,
};

#endif
