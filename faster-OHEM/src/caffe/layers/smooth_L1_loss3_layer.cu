// --------------------------------------------------------
// Fast R-CNN
// Copyright (c) Microsoft. All rights reserved.
// Written by Ross Girshick, 2015.
// Licensed under the BSD 2-clause "Simplified" license.
// See LICENSE in the Fast R-CNN project root for license
// information.
// --------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

	template <typename Dtype>
	__global__ void SmoothL1Forward3GPU(const int n, const Dtype* in, Dtype* out) {
		// f(x) = 0.5 * x^2    if |x| < 1
		//        |x| - 0.5    otherwise
		CUDA_KERNEL_LOOP(index, n) {
			Dtype val = in[index];
			Dtype abs_val = abs(val);
			if (abs_val < 1) {
				out[index] = 0.5 * val * val;
			}
			else {
				out[index] = abs_val - 0.5;
			}
		}
	}
	template <typename Dtype>
	__global__ void SmoothL1Sum3GPU(const int n, const Dtype* in, Dtype* out, const int dim) {
		CUDA_KERNEL_LOOP(index, n) {
			for (int i = 0; i < dim; i++) {
				out[index] += in[index * dim + i];
			}
		}
	}

	template <typename Dtype>
	void SmoothL1Loss3Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		//Forward_cpu(bottom, top);
		const int count = bottom[0]->count();
		const int outer_num_ = bottom[0]->num();
		const int inner_num_ = count / outer_num_;

		Dtype* top_data = top[0]->mutable_gpu_data();
		caffe_gpu_sub(
			count,
			bottom[0]->gpu_data(),
			bottom[1]->gpu_data(),
			diff_.mutable_gpu_data());    // d := b0 - b1
		if (has_weights_) {
			caffe_gpu_mul(
				count,
				bottom[2]->gpu_data(),
				diff_.gpu_data(),
				diff_.mutable_gpu_data());  // d := w * (b0 - b1)
		}
		SmoothL1Forward3GPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, diff_.gpu_data(), errors_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
		Dtype loss;
		int dim = count / bottom[0]->num();
		SmoothL1Sum3GPU<Dtype> <<<CAFFE_GET_BLOCKS(outer_num_), CAFFE_CUDA_NUM_THREADS>>>(outer_num_, errors_.gpu_data(), top_data, inner_num_);
		CUDA_POST_KERNEL_CHECK;
		int spatial_dim = diff_.height() * diff_.width();
		caffe_gpu_scale(outer_num_, 1/Dtype(outer_num_*spatial_dim), top[0]->mutable_gpu_data(),top[0]->mutable_gpu_data());
				//		int spatial_dim = diff_.height() * diff_.width();
//		//top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num(); // This is the original implementation in *Fast* R-CNN
//		top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num() / spatial_dim;  // This implementation takes effects for both RPN and Fast R-CNN.
		// For Fast R-CNN, bottom[0]->num() is 128, and spatial_dim is always 1. This implementation is equivalent.
		// For RPN, bottom[0]->num() is 1, and spatial_dim is about 1000x600/16/16 ~ 2400.
		// Also for RPN, in the SoftmaxLoss we use the default normalize version (sum of weights = batch_size = 256 ), so lambda=10 (see paper) will make SoftmaxLoss and SmoothL1Loss roughly balanced.
	}

	template <typename Dtype>
	__global__ void SmoothL1Backward3GPU(const int n, const Dtype* in, Dtype* out) {
		// f'(x) = x         if |x| < 1
		//       = sign(x)   otherwise
		CUDA_KERNEL_LOOP(index, n) {
			Dtype val = in[index];
			Dtype abs_val = abs(val);
			if (abs_val < 1) {
				out[index] = val;
			}
			else {
				out[index] = (Dtype(0) < val) - (val < Dtype(0));
			}
		}
	}

	template <typename Dtype>
	void SmoothL1Loss3Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		int count = diff_.count();
		SmoothL1Backward3GPU<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			count, diff_.gpu_data(), diff_.mutable_gpu_data());
		CUDA_POST_KERNEL_CHECK;
		for (int i = 0; i < 2; ++i) {
			if (propagate_down[i]) {
				const Dtype sign = (i == 0) ? 1 : -1;
				int spatial_dim = diff_.height() * diff_.width();
				const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num() / spatial_dim;
				caffe_gpu_axpby(
					bottom[i]->count(),              // count
					alpha,                           // alpha
					diff_.gpu_data(),                // x
					Dtype(0),                        // beta
					bottom[i]->mutable_gpu_diff());  // y
			}
		}
	}

	INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1Loss3Layer);

}  // namespace caffe
