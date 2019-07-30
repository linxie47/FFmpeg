/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "image.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern ImageMap image_map_vaapi;

const ImageMap *image_map_get_by_name(const char *name) {
    const ImageMap *m = &image_map_vaapi;
    if (!strcmp(name, m->name))
        return m;

    fprintf(stderr, "Support VAAPI image map only!\n");
    assert(0);
}

ImageMapContext *image_map_alloc(const ImageMap *image_map) {
    ImageMapContext *ret = NULL;

    if (image_map == NULL)
        return NULL;

    ret = (ImageMapContext *)malloc(sizeof(*ret));
    assert(ret);
    memset(ret, 0, sizeof(*ret));

    ret->mapper = image_map;
    if (image_map->priv_size > 0) {
        ret->priv = malloc(image_map->priv_size);
        if (!ret->priv)
            goto err;
        memset(ret->priv, 0, image_map->priv_size);
    }

    return ret;
err:
    free(ret);
    return NULL;
}

void image_map_free(ImageMapContext *context) {
    if (context == NULL)
        return;

    if (context->priv)
        free(context->priv);
    free(context);
}