#!/bin/sh

TARGET=custom
NET_NAME=deeplab
DEPLOY_MODEL_PATH=vai_q_output
TF_NETWORK_PATH=deeplab

ARCH=${TARGET}.json

vai_c_tensorflow --frozen_pb ${TF_NETWORK_PATH}/${DEPLOY_MODEL_PATH}/deploy_model.pb \
                 --arch ${ARCH} \
		 --output_dir ${TF_NETWORK_PATH}/vai_c_output_${TARGET}/ \
		 --net_name ${NET_NAME} \
		 --options "{'save_kernel':''}"


