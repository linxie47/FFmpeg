#!/bin/bash

set -e

if [ -z ${MODELS_PATH} ]; then
    echo "please set MODELS_PATH. e.g: export MODELS_PATH=/home/media/workspace/tests"
    exit 1
fi

BASEDIR=$(dirname "$0")/../..
usage="$(basename "$0") [-i <stream>] [-options] -- program to do object detection

where:
-h            show this help text
-i  <stream>  set the stream path
-s            to show on the screen
-v            to show debug log
-r  <number>  set inference request number
-b  <number>  set batch size
-d  <devices> set devices for each model(C-CPU G-GPU V-VPU H-HDDL M-MULTI O-HETERO) e.g.CGV"

if [ -z "$1" ]; then
    echo "$usage"
    exit
fi

while getopts ':b:hi:r:svd:' option; do
    case "$option" in
        h) echo "$usage"
            exit
            ;;
        i) stream=$OPTARG
            ;;
        s) show="true"
            ;;
        v) debug_log="-loglevel debug"
            ;;
        d) devices_pattern=$OPTARG
            ;;
        r) req_num=$OPTARG
            ;;
        b) batch=$OPTARG
            ;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
        *)
    esac
done
shift $((OPTIND - 1))

MODEL=mobilenet-ssd

CPU="CPU"
GPU="GPU"
VPU="MYRIAD"
HDDL="HDDL"
MULTI="MULTI"
HETERO="HETERO"
GET_DEVICE_ID() {
    case $1 in
        C)   echo $CPU;;
        G)   echo $GPU;;
        V)   echo $VPU;;
        H)   echo $HDDL;;
        M)   echo $MULTI;;
        O)   echo $HETERO;;
        *)   echo Unknown device: $1
        ;;
    esac
}

PRECISION_FP16="\"FP16\""
PRECISION_FP32="\"FP32\""
GET_PRECISION() {
    if [ -z $1 ];then
        exit 0
    fi
    case $1 in
        C)   echo $PRECISION_FP32;;
        G)   echo $PRECISION_FP16;;
        V)   echo $PRECISION_FP16;;
        H)   echo $PRECISION_FP16;;
        *)   echo $PRECISION_FP32
        ;;
    esac
}

if [ ! -z "$devices_pattern" ]; then
    DEVICE1=$(echo "${devices_pattern:0:1}")
    D_ID1=$(GET_DEVICE_ID $DEVICE1)
fi
D_ID1=${D_ID1:-$CPU}

GET_MODEL_PATH() {
    for path in ${MODELS_PATH//:/ }; do
        paths=$(find $path -name "$1*.xml" -print)
        if [ ! -z "$paths" ];
        then
            PRECISION=${2:-\"FP32\"}
            echo $(grep -l "precision=$PRECISION" $paths)
            exit 0
        fi
    done
    echo -e "\e[31mModel $1.xml file was not found. Please set MODELS_PATH\e[0m" 1>&2
    exit 1
}

DETECT_MODEL_PATH=$(GET_MODEL_PATH $MODEL $(GET_PRECISION $DEVICE1))

echo "$DETECT_MODEL_PATH"

PROC_PATH() {
    echo ${BASEDIR}/samples/model_proc/$1.json
}

req_num=${req_num:-4}
batch=${batch:-1}

ie_resize_bgr="PRE_PROCESSOR_TYPE=ie|IMAGE_FORMAT=BGR"
ie_resize_nv12="PRE_PROCESSOR_TYPE=ie|IMAGE_FORMAT=NV12"

if [ ! -z "$show" ]; then
    $BASEDIR/ffplay $debug_log -i $stream -sync video -vf \
        "format=nv12,ie_detect=model=$DETECT_MODEL_PATH:device=$D_ID1:nireq=$req_num:batch_size=$batch:configs=$ie_resize_nv12, \
        ocv_overlay"
else
    $BASEDIR/ffmpeg $debug_log -i $stream -vf \
    "format=bgr24,ie_detect=model=$DETECT_MODEL_PATH:model_proc=$(PROC_PATH $MODEL):device=${D_ID1}:nireq=$req_num:batch_size=$batch:configs=$ie_resize_bgr" \
        -y -f iemetadata /tmp/obj_detect.json
fi

