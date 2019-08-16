#!/bin/bash
# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e

BASEDIR=$(dirname "$0")/../..
echo ${PWD}/$BASEDIR

source $BASEDIR/scripts/setlocale.sh
#import GET_MODEL_PATH and PROC_PATH
source $BASEDIR/scripts/path_extractor.sh

FILE=${1}
GALLERY=$BASEDIR/scripts/gallery_generator/gallery.json

DETECTION_MODEL=face-detection-adas-0001-int8
DETECT_MODEL_PATH=$(GET_MODEL_PATH $DETECTION_MODEL)

IDENTIFICATION_MODEL=face-reidentification-retail-0095
IDENTIFICATION_MODEL_PATH=$(GET_MODEL_PATH $IDENTIFICATION_MODEL )

IDENTIFICATION_MODEL_PROC=face-reidentification-retail-0095
IDENTIFICATION_MODEL_PROC_PATH=$(PROC_PATH $IDENTIFICATION_MODEL_PROC)

DEVICE=CPU
CONFIGS="CPU_THROUGHPUT_STREAMS=24\,CPU_THREADS_NUM=96"

echo Running sample with the following parameters:
echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

ffmpeg -i ${FILE} -vf " \
  ie_detect=model=$DETECT_MODEL_PATH:device=$DEVICE:nireq=18:configs=$CONFIGS, \
  ie_classify=model=$IDENTIFICATION_MODEL_PATH:model_proc=$IDENTIFICATION_MODEL_PROC_PATH:device=$DEVICE:nireq=14, \
  identify=gallery=$GALLERY" \
  -an -f null -