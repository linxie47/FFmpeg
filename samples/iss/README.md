# iss pipelines

# Requirements
1. Git pull the latest code and build the Docker image
2. Download models using open_model_zoo, [convert](https://gitlab.devtools.intel.com/video-analytics/gstreamer-plugins/wikis/Getting-Started-Guide-%5Bdevelop%5D#download-models) and optimize the necessary models (mobilenet-ssd and resnet-50) [guide](https://gitlab.devtools.intel.com/video-analytics/gstreamer-plugins/wikis/Getting-Started-Guide-%5Bdevelop%5D#advanced-usage)
3. [Run container](https://gitlab.devtools.intel.com/video-analytics/wiki/blob/master/Getting-Started-Guide-%5Bdevelop%5D.md#run-docker-image) with mount iss samples **-v ~/samples/iss/:/root/iss**
4. Change directory to the directory with iss samples and run shell scripts (**with the specified video**) from CLX folder (for cascade lake CPU) or VCAA folder (for VCAA card)

Example: ./license_plate_recognition.sh /root/video-examples/long_Pexels_Videos_4786.h264

### For face_recognition.sh sample
For this sample you need to make the tensors of faces. The instruction is in the folder: iss/scripts/gallery_generator
