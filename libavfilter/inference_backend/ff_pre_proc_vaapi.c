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
#include <va/va.h>
#include <va/va_vpp.h>

#define VA_CALL(_FUNC)                                     \
    {                                                      \
        VAStatus _status = _FUNC;                          \
        if (_status != VA_STATUS_SUCCESS)                  \
        {                                                  \
            printf(#_FUNC " failed, sts = %d (%s).\n",     \
                    _status, vaErrorStr(_status));         \
            return;                                     \
        }                                                  \
    }

enum AVColorSpace {
    AVCOL_SPC_RGB         = 0,  ///< order of coefficients is actually GBR, also IEC 61966-2-1 (sRGB)
    AVCOL_SPC_BT709       = 1,  ///< also ITU-R BT1361 / IEC 61966-2-4 xvYCC709 / SMPTE RP177 Annex B
    AVCOL_SPC_UNSPECIFIED = 2,
    AVCOL_SPC_RESERVED    = 3,
    AVCOL_SPC_FCC         = 4,  ///< FCC Title 47 Code of Federal Regulations 73.682 (a)(20)
    AVCOL_SPC_BT470BG     = 5,  ///< also ITU-R BT601-6 625 / ITU-R BT1358 625 / ITU-R BT1700 625 PAL & SECAM / IEC 61966-2-4 xvYCC601
    AVCOL_SPC_SMPTE170M   = 6,  ///< also ITU-R BT601-6 525 / ITU-R BT1358 525 / ITU-R BT1700 NTSC
    AVCOL_SPC_SMPTE240M   = 7,  ///< functionally identical to above
    AVCOL_SPC_YCGCO       = 8,  ///< Used by Dirac / VC-2 and H.264 FRext, see ITU-T SG16
    AVCOL_SPC_YCOCG       = AVCOL_SPC_YCGCO,
    AVCOL_SPC_BT2020_NCL  = 9,  ///< ITU-R BT2020 non-constant luminance system
    AVCOL_SPC_BT2020_CL   = 10, ///< ITU-R BT2020 constant luminance system
    AVCOL_SPC_SMPTE2085   = 11, ///< SMPTE 2085, Y'D'zD'x
    AVCOL_SPC_CHROMA_DERIVED_NCL = 12, ///< Chromaticity-derived non-constant luminance system
    AVCOL_SPC_CHROMA_DERIVED_CL = 13, ///< Chromaticity-derived constant luminance system
    AVCOL_SPC_ICTCP       = 14, ///< ITU-R BT.2100-0, ICtCp
    AVCOL_SPC_NB                ///< Not part of ABI
};

#define II_MAX(a, b) ((a) > (b) ? (a) : (b))
#define II_MIN(a, b) ((a) > (b) ? (b) : (a))

#if CONFIG_VAAPI

struct _VAAPIVpp {
    VADisplay display;
    VASurfaceID va_surface;
    VAConfigID va_config;
    VAContextID va_context;
    VAImageFormat *format_list; //!< Surface formats which can be used with this device.
    int nb_formats;
    int scale_w, scale_h;
    VAImage va_image;
    VAImageFormat va_format_selected;
};
typedef struct _VAAPIVpp VAAPIVpp;

typedef struct FFPreProc {
    VAAPIVpp *vaapi_vpp;
} FFPreProc;

static int vaapi_vpp_colour_standard(enum AVColorSpace av_cs)
{
    switch(av_cs) {
#define CS(av, va) case AVCOL_SPC_ ## av: return VAProcColorStandard ## va;
        CS(BT709,     BT709);
        CS(BT470BG,   BT601);
        CS(SMPTE170M, None);
        CS(SMPTE240M, None);
#undef CS
    default:
        return VAProcColorStandardNone;
    }
}

static int va_surface_alloc(VAAPIVpp *va_vpp, size_t width, size_t height, int format)
{
    int i, rt_format, fourcc;
    VASurfaceAttrib surface_attrib;

    if (!va_vpp)
        return -1;

    if (format == 0 || format == FOURCC_BGRX) {
        fourcc = VA_FOURCC_BGRX; rt_format = VA_RT_FORMAT_RGB32;
    } else if (format == FOURCC_RGBP) {
        fourcc = VA_FOURCC_RGBP; rt_format = VA_RT_FORMAT_RGBP;
    } else
        return -1;

    surface_attrib.type          = VASurfaceAttribPixelFormat;
    surface_attrib.flags         = VA_SURFACE_ATTRIB_SETTABLE;
    surface_attrib.value.type    = VAGenericValueTypeInteger;
    surface_attrib.value.value.i = fourcc;

    VA_CALL(vaCreateConfig(va_vpp->display, VAProfileNone,
                           VAEntrypointVideoProc, 0, 0, &va_vpp->va_config));

    VA_CALL(vaCreateSurfaces(va_vpp->display, rt_format, width, height,
                             &va_vpp->va_surface, 1, &surface_attrib, 1));

    VA_CALL(vaCreateContext(va_vpp->display, va_vpp->va_config,
                            width, height, VA_PROGRESSIVE,
                            &va_vpp->va_surface, 1, &va_vpp->va_context));

    for (i = 0; i < va_vpp->nb_formats; i++) {
        if (va_vpp->format_list[i].fourcc == fourcc) {
            va_vpp->va_format_selected = va_vpp->format_list[i];
            break;
        }
    }

    return VA_STATUS_SUCCESS;
}

static int FFPreProcInit(PreProcContext *context, void *priv)
{
    FFPreProc *ff_pre_proc = (FFPreProc *)context->priv;
    VAAPIVpp *va_vpp = malloc(sizeof(VAAPIVpp));
    VADisplay display = NULL;
    VAImageFormat *image_list = NULL;
    VAStatus vas;
    int err, image_count;

    ff_pre_proc->vaapi_vpp = va_vpp;

    display = (VADisplay)priv;

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

    va_vpp->display = display;
    va_vpp->format_list = image_list;
    va_vpp->nb_formats = image_count;
    va_vpp->va_config = VA_INVALID_ID;
    va_vpp->va_context = VA_INVALID_ID;
    va_vpp->va_surface = VA_INVALID_ID;

    return VA_STATUS_SUCCESS;
fail:
    if (image_list)
        free(image_list);
    return err;
}

static void FFPreProcConvert(PreProcContext *context, const Image *src, Image *dst, int bAllocateDestination)
{
    FFPreProc *ff_pre_proc = (FFPreProc *)context->priv;
    VAAPIVpp *va_vpp = ff_pre_proc->vaapi_vpp;
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
    if (src->format == dst->format && src->format == FOURCC_RGBP &&
        src->width == dst->width &&
        src->height == dst->height) {
        int planes_count = GetPlanesCount(src->format);
        // RGB->BGR
        Image src_bgr = *src;
        src_bgr.planes[0] = src->planes[2];
        src_bgr.planes[2] = src->planes[0];
        for (int i = 0; i < planes_count; i++) {
            if (src_bgr.width == src_bgr.stride[i]) {
                memcpy(dst->planes[i], src_bgr.planes[i],
                        src_bgr.width * src_bgr.height * sizeof(uint8_t));
            } else {
                int dst_stride = dst->stride[i] * sizeof(uint8_t);
                int src_stride = src_bgr.stride[i] * sizeof(uint8_t);
                for (int r = 0; r < src_bgr.height; r++) {
                    memcpy(dst->planes[i] + r * dst_stride,
                            src_bgr.planes[i] + r * src_stride, dst->width);
                }
            }
        }
        return;
    }

    input_surface = (VASurfaceID)(uintptr_t)src->surface_id;

    if (va_vpp->va_surface == VA_INVALID_ID ||
            va_vpp->scale_w != dst->width || va_vpp->scale_h != dst->height) {
        if (va_vpp->va_surface != VA_INVALID_ID) {
            vas = vaDestroySurfaces(va_vpp->display, &va_vpp->va_surface, 1);
            if (vas != VA_STATUS_SUCCESS) {
                fprintf(stderr, "Failed to destroy surface, %s\n", vaErrorStr(vas));
            }
        }

        va_vpp->scale_w = dst->width;
        va_vpp->scale_h = dst->height;
        ret = va_surface_alloc(ff_pre_proc->vaapi_vpp, dst->width, dst->height, dst->format);
        if (ret < 0) {
            fprintf(stderr,"Create va surface failed\n");
            return;
        }
    }
    output_surface = ff_pre_proc->vaapi_vpp->va_surface;

    rect_x = II_MAX(src->rect.x, 0);
    rect_y = II_MAX(src->rect.y, 0);
    rect_width = II_MIN(src->rect.width - (rect_x - src->rect.x), src->width - rect_x);
    rect_height = II_MIN(src->rect.height - (rect_y - src->rect.y), src->height - rect_y);

    input_region = (VARectangle) {
        .x      = rect_x,
        .y      = rect_y,
        .width  = rect_width,
        .height = rect_height,
    };

    memset(&params, 0, sizeof(params));

    params.surface = input_surface;
    params.surface_region = &input_region;
    params.surface_color_standard =
        vaapi_vpp_colour_standard(src->colorspace);

    params.output_region = 0;
    params.output_background_color = 0xff000000;
    params.output_color_standard = params.surface_color_standard;

    params.pipeline_flags = 0;
    params.filter_flags = VA_FILTER_SCALING_DEFAULT;

    scale_w = dst->width;
    scale_h = dst->height;

    VA_CALL(vaBeginPicture(va_vpp->display, va_vpp->va_context, output_surface));

    VA_CALL(vaCreateBuffer(va_vpp->display, va_vpp->va_context,
                           VAProcPipelineParameterBufferType,
                           sizeof(params), 1, &params, &params_id));

    VA_CALL(vaRenderPicture(va_vpp->display, va_vpp->va_context,
                            &params_id, 1));

    VA_CALL(vaEndPicture(va_vpp->display, va_vpp->va_context));

    VA_CALL(vaDestroyBuffer(va_vpp->display, params_id));

    VA_CALL(vaSyncSurface(va_vpp->display, output_surface));

    va_image_ptr = &va_vpp->va_image;

    VA_CALL(vaCreateImage(va_vpp->display, &va_vpp->va_format_selected,
                          scale_w, scale_h, va_image_ptr));

    VA_CALL(vaGetImage(va_vpp->display, output_surface,
                       0, 0, scale_w, scale_h,
                       va_image_ptr->image_id));

    VA_CALL(vaMapBuffer(va_vpp->display, va_image_ptr->buf, &address));

    dst_rgb.planes[0] = dst->planes[2];
    dst_rgb.planes[2] = dst->planes[0];
    for (i = 0; i < II_MIN(va_image_ptr->num_planes, MAX_PLANES_NUMBER); i++) {
        int stride = va_image_ptr->pitches[i];
        src_data = (uint8_t *)address + va_image_ptr->offsets[i];
        for (j = 0; j < scale_h; j++) {
            memcpy(dst_rgb.planes[i] + j * scale_w, src_data + j * stride, scale_w);
        }
    }

back:
    VA_CALL(vaUnmapBuffer(va_vpp->display, va_image_ptr->buf));

    return;
}

static void FFPreProcDestroy(PreProcContext *context)
{
    FFPreProc *ff_pre_proc = (FFPreProc *)context->priv;
    VAStatus vas;
    VAAPIVpp *va_vpp = ff_pre_proc->vaapi_vpp;

    if (ff_pre_proc->vaapi_vpp) {
        if (va_vpp->va_surface != VA_INVALID_ID) {
            vas = vaDestroySurfaces(va_vpp->display, &va_vpp->va_surface, 1);
            if (vas != VA_STATUS_SUCCESS) {
                fprintf(stderr, "Failed to destroy surface, %s\n", vaErrorStr(vas));
            }
        }
        if (va_vpp->va_context != VA_INVALID_ID) {
            vaDestroyContext(va_vpp->display, va_vpp->va_context);
            va_vpp->va_context = VA_INVALID_ID;
        }

        if (va_vpp->va_config != VA_INVALID_ID) {
            vaDestroyConfig(va_vpp->display, va_vpp->va_config);
            va_vpp->va_config = VA_INVALID_ID;
        }

        free(va_vpp->format_list);
        return;
    }

    return;
}

PreProc pre_proc_vaapi = {
    .name = "vaapi",
    .priv_size = sizeof(FFPreProc),
    .mem_type = MEM_TYPE_VAAPI,
    .Init = FFPreProcInit,
    .Convert = FFPreProcConvert,
    .Destroy = FFPreProcDestroy,
};

#endif
