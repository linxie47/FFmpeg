#!/bin/bash
# ==============================================================================
# Copyright (C) 2018-2019 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

set -e

BASEDIR=$(dirname "$0")/../..
source $BASEDIR/scripts/setlocale.sh
#import GET_MODEL_PATH and PROC_PATH
source $BASEDIR/scripts/path_extractor.sh

FILE=${1}

DETECTION_MODEL=vehicle-detection-adas-0002-int8
DETECTION_MODEL_PATH=$(GET_MODEL_PATH $DETECTION_MODEL )

CLASSIFY_MODEL=vehicle-attributes-recognition-barrier-0039-int8
CLASSIFY_MODEL_PATH=$(GET_MODEL_PATH $CLASSIFY_MODEL )

DETECTION_MODEL_PROC=vehicle-detection-adas-0002
DETECTION_MODEL_PROC_PATH=$(PROC_PATH $DETECTION_MODEL_PROC)

CLASSIFY_MODEL_PROC=vehicle-attributes-recognition-barrier-0039
CLASSIFY_MODEL_PROC_PATH=$(PROC_PATH $CLASSIFY_MODEL_PROC)

DEVICE=CPU
CONFIGS="CPU_THROUGHPUT_STREAMS=24\,CPU_THREADS_NUM=96"

echo Running sample with the following parameters:
echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

ffmpeg -i ${FILE} -vf " \
  ie_detect=model=$DETECTION_MODEL_PATH:model_proc=$DETECTION_MODEL_PROC_PATH:device=$DEVICE:nireq=16:configs=$CONFIGS, \
  ie_classify=model=$CLASSIFY_MODEL_PATH:model_proc=$CLASSIFY_MODEL_PROC_PATH:device=$DEVICE:nireq=14" \
  -an -f null -