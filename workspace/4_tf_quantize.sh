#!/bin/sh

TF_NETWORK_PATH=deeplab
vai_q_tensorflow quantize --input_frozen_graph ${TF_NETWORK_PATH}/float/frozen_model.pb \
			  --input_fn example_file.input_fn.calib_input \
			  --output_dir ${TF_NETWORK_PATH}/vai_q_output \
	                  --input_nodes Input \
			  --output_nodes output_logit \
			  --input_shapes ?,384,384,3 \
			  --calib_iter 1000 \
