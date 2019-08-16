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

DETECTION_MODEL=vehicle-license-plate-detection-barrier-0106-fp16
DETECTION_MODEL_PATH=$(GET_MODEL_PATH $DETECTION_MODEL )

CLASSIFY_MODEL=license-plate-recognition-barrier-0001-fp16
CLASSIFY_MODEL_PATH=$(GET_MODEL_PATH $CLASSIFY_MODEL )

DETECTION_MODEL_PROC=vehicle-license-plate-detection-barrier-0106
DETECTION_MODEL_PROC_PATH=$(PROC_PATH $DETECTION_MODEL_PROC) 

CLASSIFY_MODEL_PROC=license-plate-recognition-barrier-0001
CLASSIFY_MODEL_PROC_PATH=$(PROC_PATH $CLASSIFY_MODEL_PROC)

DEVICE=HDDL

HW_ACCEL="-flags unaligned -hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"

echo Running sample with the following parameters:
echo LD_LIBRARY_PATH=${LD_LIBRARY_PATH}

ffmpeg $HW_ACCEL -i ${FILE} -vf " \
  ie_detect=model=$DETECTION_MODEL_PATH:model_proc=$DETECTION_MODEL_PROC_PATH:device=$DEVICE:nireq=24, \
  ie_classify=model=$CLASSIFY_MODEL_PATH:model_proc=$CLASSIFY_MODEL_PROC_PATH:device=$DEVICE:nireq=24" \
  -an -f null -
