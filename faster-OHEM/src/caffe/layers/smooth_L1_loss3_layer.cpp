// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
void SmoothL1Loss3Layer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_weights_ = (bottom.size() == 3);
}

template <typename Dtype>
void SmoothL1Loss3Layer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  //LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  if (has_weights_) {
    CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
    CHECK_EQ(bottom[0]->height(), bottom[2]->height());
    CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  }
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  errors_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  top[0]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void SmoothL1Loss3Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
	int count = bottom[0]->count();
	const int outer_num_ = bottom[0]->num();
	const int inner_num_ = count / outer_num_;
	for (int i = 0; i < count; i++) {
		diff_.mutable_cpu_data()[i] = bottom[0]->cpu_data()[i] 
								- bottom[1]->cpu_data()[i];
		if (has_weights_) {
			diff_.mutable_cpu_data()[i] = bottom[2]->cpu_data()[i] 
								* diff_.cpu_data()[i];
		}
	}

	for (int i = 0; i < outer_num_; i++) {
		Dtype loss = 0;
		for ( int j = 0; j < inner_num_; j++){
			Dtype val = diff_.cpu_data()[i*inner_num_+j];
			Dtype abs_val = abs(val);
				if (abs_val < 1) {
					loss += 0.5 * val * val;		
				} else {
					loss += abs_val - 0.5;	
				}
		}
		top[0]->mutable_cpu_data()[i] = loss;
	}
  //NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1Loss3Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1Loss3Layer);
#endif

INSTANTIATE_CLASS(SmoothL1Loss3Layer);
REGISTER_LAYER_CLASS(SmoothL1Loss3);

}  // namespace caffe
