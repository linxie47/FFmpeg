#!/bin/bash

set -e

if [ -z ${MODELS_PATH} ]; then
    echo "please set MODELS_PATH. e.g: export MODELS_PATH=/home/media/workspace/tests"
    exit 1
fi

BASEDIR=$(dirname "$0")/../..
usage="$(basename "$0") [-i <stream>] [-options] -- vehicle/license plate recognition

where:
-h            show this help text
-a            use hardware decode to accelerate
-i  <stream>  set the stream path
-s            to show on the screen
-v            to show debug log
-r  <number>  set inference request number
-b  <number>  set batch size
-d  <devices> set devices for each model(C-CPU G-GPU V-VPU H-HDDL) e.g.CGV
-m            set method for json output
-c            set converter type:all-to-json,detection-to-json,classification-to-json,tensor-to-file(only for reid tensor generator)
-j            set the json file path"


if [ -z "$1" ]; then
    echo "$usage"
    exit
fi

while getopts ':ab:hi:r:svd:m:c:j:' option; do
    case "$option" in
        h) echo "$usage"
            exit
            ;;
        a) hw_accel="-flags unaligned -hwaccel vaapi -hwaccel_output_format vaapi -hwaccel_device /dev/dri/renderD128"
           hw_dl="hwdownload,format=bgr0,"
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
        m) json_method=$OPTARG
            ;;
        c) json_convert=$OPTARG
            ;;
        j) json_path=$OPTARG
            ;;
        \?) printf "illegal option: -%s\n" "$OPTARG" >&2
            echo "$usage" >&2
            exit 1
            ;;
        *)
    esac
done
shift $((OPTIND - 1))

MODEL1=vehicle-license-plate-detection-barrier-0106
MODEL2=vehicle-attributes-recognition-barrier-0039
MODEL3=license-plate-recognition-barrier-0001
ALL=all-to-json
METHOD=output
JSON_FILE_PATH=/tmp/

CPU="CPU"
GPU="GPU"
VPU="MYRIAD"
HDDL="HDDL"
GET_DEVICE_ID() {
    case $1 in
        C)   echo $CPU;;
        G)   echo $GPU;;
        V)   echo $VPU;;
        H)   echo $HDDL;;
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
        *)   echo Unknown device: $1
        ;;
    esac
}

if [ ! -z "$devices_pattern" ]; then
    DEVICE1=$(echo "${devices_pattern:0:1}")
    DEVICE2=$(echo "${devices_pattern:1:1}")
    DEVICE3=$(echo "${devices_pattern:2:1}")
    D_ID1=$(GET_DEVICE_ID $DEVICE1)
    D_ID2=$(GET_DEVICE_ID $DEVICE2)
    D_ID3=$(GET_DEVICE_ID $DEVICE3)
fi
D_ID1=${D_ID1:-$CPU}
D_ID2=${D_ID2:-$CPU}
D_ID3=${D_ID3:-$CPU}

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

DETECT_MODEL_PATH=$(GET_MODEL_PATH $MODEL1 $(GET_PRECISION $DEVICE1))
CLASS_MODEL_PATH=$(GET_MODEL_PATH  $MODEL2 $(GET_PRECISION $DEVICE2))
CLASS_MODEL_PATH1=$(GET_MODEL_PATH $MODEL3 $(GET_PRECISION $DEVICE3))

echo "$DETECT_MODEL_PATH"
echo "$CLASS_MODEL_PATH"
echo "$CLASS_MODEL_PATH1"

PROC_PATH() {
    echo ${BASEDIR}/samples/model_proc/$1.json
}

req_num1=${req_num:-4}
req_num2=${req_num:-4}
req_num3=${req_num:-4}
batch=${batch:-1}
json_convert=${json_convert:-$ALL}
json_method=${json_method:-$METHOD}
json_path=${json_path:-$JSON_FILE_PATH}

if [ ! -z "$show" ]; then
    $BASEDIR/ffplay $debug_log -i $stream -sync video -vf \
        "ie_detect=model=$DETECT_MODEL_PATH:model_proc=$(PROC_PATH $MODEL1):device=$D_ID1, \
        ie_classify=model=$CLASS_MODEL_PATH:model_proc=$(PROC_PATH $MODEL2):device=$D_ID2, \
        ie_classify=model=$CLASS_MODEL_PATH1:model_proc=$(PROC_PATH $MODEL3):device=$D_ID3, \
        ocv_overlay"
else
    $BASEDIR/ffmpeg $debug_log $hw_accel \
        -i $stream -vf \
        "${hw_dl}ie_detect=model=$DETECT_MODEL_PATH:model_proc=$(PROC_PATH $MODEL1):device=$D_ID1:nireq=$req_num1:batch_size=$batch, \
        ie_classify=model=$CLASS_MODEL_PATH:model_proc=$(PROC_PATH $MODEL2):device=$D_ID2:nireq=$req_num2:batch_size=$batch, \
        ie_classify=model=$CLASS_MODEL_PATH1:model_proc=$(PROC_PATH $MODEL3):device=$D_ID3:nireq=$req_num3:batch_size=$batch, \
        metaconvert=converter=$json_convert:method=$json_method:location=$json_path" \
        -an -f null - #-f iemetadata -y /tmp/security.json
fi
