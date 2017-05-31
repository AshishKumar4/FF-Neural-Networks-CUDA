#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"
#include "sm_35_atomic_functions.h"

#include "idx.h"
#include "neuron.h"

__global__ void Input_Layer_Thread(neuron* n, idx_content_img* im, int index);
__global__ void Input_Layer_Thread_vec(neuron* n, int* in_vec, int in_sz, int index);

__global__ void Output_Layer_SumGenerator_Thread(neuron* n);

__global__ void Output_Layer_SoftmaxSum_Thread(neuron* n, int out_sz, int o_l_sz_r2, double* tmp);

__global__ void Output_Layer_Thread(neuron* n, uint8_t* values, int index, double* tmp);
__global__ void Output_Layer_Thread_vec(neuron* n, int* out_vec, int out_sz, int index, double* tmp);

__global__ void Output_Layer_Test_Thread(neuron* n, uint8_t* values, int index, int* output);
__global__ void Output_Layer_Test_Thread_vec(neuron* n, int* out_vec, int out_sz, int index, int* output);

__global__ void dot_product_FeedForward(neuron* n, int r_b);

__global__ void dot_product_BackwardPropogation(neuron* n);

__global__ void ForwardPropogation(neuron* n);


__global__ void BackPropogation(neuron* n);

__global__ void DeltaWeightPropogation(neuron* n);

__global__ void linker_thread(neuron* n, int n_no, neuron* in, int in_no);