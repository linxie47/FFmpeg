/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "pre_proc.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if CONFIG_VAAPI
#include <va/va.h>
#include <va/va_vpp.h>
#endif

#define VA_CALL(_FUNC)                                                                                                 \
    {                                                                                                                  \
        VAStatus _status = _FUNC;                                                                                      \
        if (_status != VA_STATUS_SUCCESS) {                                                                            \
            printf(#_FUNC " failed, sts = %d (%s).\n", _status, vaErrorStr(_status));                                  \
            return;                                                                                                    \
        }                                                                                                              \
    }

#define VA_CALL_RET(_FUNC)                                                                                             \
    {                                                                                                                  \
        VAStatus _status = _FUNC;                                                                                      \
        if (_status != VA_STATUS_SUCCESS) {                                                                            \
            printf(#_FUNC " failed, sts = %d (%s).\n", _status, vaErrorStr(_status));                                  \
            return _status;                                                                                            \
        }                                                                                                              \
    }

enum AVColorSpace {
    AVCOL_SPC_RGB = 0,   ///< order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
    AVCOL_SPC_BT709 = 1, ///< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
    AVCOL_SPC_UNSPECIFIED = 2,
    AVCOL_SPC_RESERVED = 3,
    AVCOL_SPC_FCC = 4, ///< FCC Title 47 Code of Federal Regulations 73.682 (a)(20)
    AVCOL_SPC_BT470BG =
        5, ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
    AVCOL_SPC_SMPTE170M = 6, ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    AVCOL_SPC_SMPTE240M = 7, ///< functionally identical to above
    AVCOL_SPC_YCGCO = 8,     ///< Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
    AVCOL_SPC_YCOCG = AVCOL_SPC_YCGCO,
    AVCOL_SPC_BT2020_NCL = 9,          ///< ITU-R BT2020 non-constant luminance system
    AVCOL_SPC_BT2020_CL = 10,          ///< ITU-R BT2020 constant luminance system
    AVCOL_SPC_SMPTE2085 = 11,          ///< SMPTE 2085, Y'D'zD'x
    AVCOL_SPC_CHROMA_DERIVED_NCL = 12, ///< Chromaticity-derived non-constant luminance system
    AVCOL_SPC_CHROMA_DERIVED_CL = 13,  ///< Chromaticity-derived constant luminance system
    AVCOL_SPC_ICTCP = 14,              ///< ITU-R BT.2100-0, ICtCp
    AVCOL_SPC_NB                       ///< Not part of ABI
};

#define II_MAX(a, b) ((a) > (b) ? (a) : (b))
#define II_MIN(a, b) ((a) > (b) ? (b) : (a))

#if CONFIG_VAAPI

typedef struct _VAAPIPreProc {
    VADisplay display;
    VASurfaceID va_surface;
    VAConfigID va_config;
    VAContextID va_context;
    VAImageFormat *format_list; //!< Surface formats which can be used with this device.
    int nb_formats;
    int scale_w, scale_h;
    VAImage va_image;
    VAImageFormat va_format_selected;
} VAAPIPreProc;

static int vaapi_vpp_colour_standard(enum AVColorSpace av_cs) {
    switch (av_cs) {
#define CS(av, va)                                                                                                     \
    case AVCOL_SPC_##av:                                                                                               \
        return VAProcColorStandard##va;
        CS(BT709, BT709);
        CS(BT470BG, BT601);
        CS(SMPTE170M, None);
        CS(SMPTE240M, None);
#undef CS
    default:
        return VAProcColorStandardNone;
    }
}

static int va_surface_alloc(VAAPIPreProc *vaapi_pre_proc, size_t width, size_t height, int format) {
    int i, rt_format, fourcc;
    VASurfaceAttrib surface_attrib;

    if (format == 0 || format == FOURCC_BGRX) {
        fourcc = VA_FOURCC_BGRX;
        rt_format = VA_RT_FORMAT_RGB32;
    } else if (format == FOURCC_RGBP) {
        fourcc = VA_FOURCC_RGBP;
        rt_format = VA_RT_FORMAT_RGBP;
    } else
        return -1;

    surface_attrib.type = VASurfaceAttribPixelFormat;
    surface_attrib.flags = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = fourcc;

    VA_CALL_RET(vaCreateConfig(vaapi_pre_proc->display, VAProfileNone, VAEntrypointVideoProc, 0, 0,
                               &vaapi_pre_proc->va_config));

    VA_CALL_RET(vaCreateSurfaces(vaapi_pre_proc->display, rt_format, width, height, &vaapi_pre_proc->va_surface, 1,
                                 &surface_attrib, 1));

    VA_CALL_RET(vaCreateContext(vaapi_pre_proc->display, vaapi_pre_proc->va_config, width, height, VA_PROGRESSIVE,
                                &vaapi_pre_proc->va_surface, 1, &vaapi_pre_proc->va_context));

    for (i = 0; i < vaapi_pre_proc->nb_formats; i++) {
        if (vaapi_pre_proc->format_list[i].fourcc == fourcc) {
            vaapi_pre_proc->va_format_selected = vaapi_pre_proc->format_list[i];
            break;
        }
    }

    return VA_STATUS_SUCCESS;
}

static int FFPreProcInit(PreProcContext *context, void *priv) {
    VAAPIPreProc *vaapi_pre_proc = (VAAPIPreProc *)context->priv;
    VADisplay display = (VADisplay)priv;
    VAImageFormat *image_list = NULL;
    VAStatus vas;
    int err, image_count;

    image_count = vaMaxNumImageFormats(display);
    if (image_count <= 0) {
        err = -1;
        goto fail;
    }
    image_list = malloc(image_count * sizeof(*image_list));
    if (!image_list) {
        err = -1;
        goto fail;
    }
    vas = vaQueryImageFormats(display, image_list, &image_count);
    if (vas != VA_STATUS_SUCCESS) {
        err = -1;
        goto fail;
    }

    vaapi_pre_proc->display = display;
    vaapi_pre_proc->format_list = image_list;
    vaapi_pre_proc->nb_formats = image_count;
    vaapi_pre_proc->va_config = VA_INVALID_ID;
    vaapi_pre_proc->va_context = VA_INVALID_ID;
    vaapi_pre_proc->va_surface = VA_INVALID_ID;

    return VA_STATUS_SUCCESS;
fail:
    if (image_list)
        free(image_list);
    return err;
}

static void FFPreProcConvert(PreProcContext *context, const Image *src, Image *dst, int bAllocateDestination) {
    VAAPIPreProc *vaapi_pre_proc = (VAAPIPreProc *)context->priv;
    VAImage *va_image_ptr;
    VABufferID params_id;
    VASurfaceID input_surface, output_surface;
    VAProcPipelineParameterBuffer params;
    VARectangle input_region;
    void *address = NULL;
    int rect_x, rect_y, rect_width, rect_height, scale_w, scale_h, i, j;
    uint8_t *src_data;
    Image dst_rgb = *dst;
    VAStatus vas;
    int ret;

    // identical format and resolution
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

    input_surface = (VASurfaceID)(uintptr_t)src->surface_id;

    if (vaapi_pre_proc->va_surface == VA_INVALID_ID || vaapi_pre_proc->scale_w != dst->width ||
        vaapi_pre_proc->scale_h != dst->height) {
        if (vaapi_pre_proc->va_surface != VA_INVALID_ID) {
            vas = vaDestroySurfaces(vaapi_pre_proc->display, &vaapi_pre_proc->va_surface, 1);
            if (vas != VA_STATUS_SUCCESS) {
                fprintf(stderr, "Failed to destroy surface, %s\n", vaErrorStr(vas));
            }
        }

        vaapi_pre_proc->scale_w = dst->width;
        vaapi_pre_proc->scale_h = dst->height;
        ret = va_surface_alloc(vaapi_pre_proc, dst->width, dst->height, dst->format);
        if (ret < 0) {
            fprintf(stderr, "Create va surface failed\n");
            return;
        }
    }
    output_surface = vaapi_pre_proc->va_surface;

    rect_x = II_MAX(src->rect.x, 0);
    rect_y = II_MAX(src->rect.y, 0);
    rect_width = II_MIN(src->rect.width - (rect_x - src->rect.x), src->width - rect_x);
    rect_height = II_MIN(src->rect.height - (rect_y - src->rect.y), src->height - rect_y);

    input_region = (VARectangle){
        .x = rect_x,
        .y = rect_y,
        .width = rect_width,
        .height = rect_height,
    };

    memset(&params, 0, sizeof(params));

    params.surface = input_surface;
    params.surface_region = &input_region;
    params.surface_color_standard = vaapi_vpp_colour_standard(src->colorspace);

    params.output_region = 0;
    params.output_background_color = 0xff000000;
    params.output_color_standard = params.surface_color_standard;

    params.pipeline_flags = 0;
    params.filter_flags = VA_FILTER_SCALING_DEFAULT;

    scale_w = dst->width;
    scale_h = dst->height;

    VA_CALL(vaBeginPicture(vaapi_pre_proc->display, vaapi_pre_proc->va_context, output_surface));

    VA_CALL(vaCreateBuffer(vaapi_pre_proc->display, vaapi_pre_proc->va_context, VAProcPipelineParameterBufferType,
                           sizeof(params), 1, &params, &params_id));

    VA_CALL(vaRenderPicture(vaapi_pre_proc->display, vaapi_pre_proc->va_context, &params_id, 1));

    VA_CALL(vaEndPicture(vaapi_pre_proc->display, vaapi_pre_proc->va_context));

    VA_CALL(vaDestroyBuffer(vaapi_pre_proc->display, params_id));

    VA_CALL(vaSyncSurface(vaapi_pre_proc->display, output_surface));

    va_image_ptr = &vaapi_pre_proc->va_image;

    VA_CALL(
        vaCreateImage(vaapi_pre_proc->display, &vaapi_pre_proc->va_format_selected, scale_w, scale_h, va_image_ptr));

    VA_CALL(vaGetImage(vaapi_pre_proc->display, output_surface, 0, 0, scale_w, scale_h, va_image_ptr->image_id));

    VA_CALL(vaMapBuffer(vaapi_pre_proc->display, va_image_ptr->buf, &address));

    dst_rgb.planes[0] = dst->planes[2];
    dst_rgb.planes[2] = dst->planes[0];
    for (i = 0; i < II_MIN(va_image_ptr->num_planes, MAX_PLANES_NUMBER); i++) {
        int stride = va_image_ptr->pitches[i];
        src_data = (uint8_t *)address + va_image_ptr->offsets[i];
        for (j = 0; j < scale_h; j++) {
            memcpy(dst_rgb.planes[i] + j * scale_w, src_data + j * stride, scale_w);
        }
    }

    VA_CALL(vaUnmapBuffer(vaapi_pre_proc->display, va_image_ptr->buf));

    return;
}

static void FFPreProcDestroy(PreProcContext *context) {
    VAAPIPreProc *vaapi_pre_proc = (VAAPIPreProc *)context->priv;

    if (!vaapi_pre_proc)
        return;

    if (vaapi_pre_proc->va_surface != VA_INVALID_ID) {
        VAStatus vas = vaDestroySurfaces(vaapi_pre_proc->display, &vaapi_pre_proc->va_surface, 1);
        if (vas != VA_STATUS_SUCCESS) {
            fprintf(stderr, "Failed to destroy surface, %s\n", vaErrorStr(vas));
        }
    }
    if (vaapi_pre_proc->va_context != VA_INVALID_ID) {
        vaDestroyContext(vaapi_pre_proc->display, vaapi_pre_proc->va_context);
        vaapi_pre_proc->va_context = VA_INVALID_ID;
    }

    if (vaapi_pre_proc->va_config != VA_INVALID_ID) {
        vaDestroyConfig(vaapi_pre_proc->display, vaapi_pre_proc->va_config);
        vaapi_pre_proc->va_config = VA_INVALID_ID;
    }

    if (vaapi_pre_proc->format_list) {
        free(vaapi_pre_proc->format_list);
        vaapi_pre_proc->format_list = NULL;
    }
}

PreProc pre_proc_vaapi = {
    .name = "vaapi",
    .priv_size = sizeof(VAAPIPreProc),
    .mem_type = MEM_TYPE_VAAPI,
    .Init = FFPreProcInit,
    .Convert = FFPreProcConvert,
    .Destroy = FFPreProcDestroy,
};

#endif // #if CONFIG_VAAPI
