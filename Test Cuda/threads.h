#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"

#include "thrust\random\normal_distribution.h"
#include "thrust\random\linear_congruential_engine.h"

#include "sm_35_atomic_functions.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "idx.h"
#include "neuron.h"
#include "dot_adder.h"

__global__ void Input_Layer_Thread(neuron* n, idx_content_img* im, int index)
{
	int id = blockIdx.x + threadIdx.x;
	double d = 35;
	n[id].output = tanh((double)(((double)im[index].values[id] - d)) / (d*3.7));
}

__global__ void Output_Layer_SumGenerator_Thread(neuron* n)
{
	int id = blockIdx.x + threadIdx.x;
	n[id].sum = 0;
	for (int j = 0; j < n[id].in_no; j++)
	{
		n[id].sum += n[id].input_n[j]->output * n[id].input_weight[j];
	}
}

__global__ void Output_Layer_SoftmaxSummation_thread(neuron* n, double* tmp)
{
	/*Softmax Output*/
	double d = 0;
	for (int i = 0; i < 10; i++)
	{
		d += powf(12, n[i].sum);
	}
	*tmp = d;
}

__global__ void Output_Layer_SoftmaxSum_Thread(neuron* n, int out_sz, double* tmp)
{
	double* d_sum_16 = dot_adders[3].dot_sum;
	if(threadIdx.x < out_sz)
		d_sum_16[threadIdx.x] = powf(12, n[threadIdx.x].sum);
	else d_sum_16[threadIdx.x] = 0;
	//__syncthreads();
	if (threadIdx.x == 0)
	{
		int tm = 16;
		for (int i = 0; i < 5; i++)
		{
			tm /= 2;
			dot_adder << <1, tm >> > (4, tm);	// lg16 = 4
			//__syncthreads();
		}
		*tmp = d_sum_16[0];
	}
}

__global__ void Output_Layer_Thread(neuron* n, uint8_t* values, int index, double* tmp)
{
	int digit = values[index];
	double d = *tmp;
	int id = blockIdx.x + threadIdx.x;
	n[id].output = powf(12, n[id].sum) / d;
	if (id == digit)
	{
		n[id].error = n[id].output - 1;
	}
	else
	{
		n[id].error = n[id].output;
	}
}

__global__ void Output_Layer_Test_Thread(neuron* n, uint8_t* values, int index, int* output)
{
	int digit = values[index];
	int _i = 0;
	double opp = 0;
	/*Softmax Output*/
	double d = 0;
	for (int i = 0; i < 10; i++)
	{
		n[i].sum = 0;
		for (int j = 0; j < n[i].in_no; j++)
		{
			n[i].sum += n[i].input_n[j]->output * n[i].input_weight[j];
		}
		d += powf(12, n[i].sum);
	}
	for (int i = 0; i < 10; i++)
	{
		n[i].output = powf(12, n[i].sum) / d;
		
		if (opp < n[i].output)
		{
			opp = n[i].output;
			_i = i;
		}
	}

	if (_i == digit)
	{
		printf("%d=>", _i);
		*output = *output + 1;
	}
	else printf("\n");
}

__global__ void dot_product_FeedForward(neuron* n)
{
	int id = threadIdx.x;
	neuron* self = &n[blockIdx.x];
	//int base = (int)ceilf(log2f(self->in_no));
	//printf("->%d", base);
	__shared__ double d[1024];
	if (id < self->in_no)
		d[id] = self->input_weight[id] * self->input_n[id]->output;
	else d[id] = 0;
	__syncthreads();
	if (id == 0)
	{
		for (int i = 1; i < 1024; i++)
		{
			d[0] += d[i];
		}
		self->sum = d[0];
		//if (self->in_no > 0)
		{
			self->output = tanh(self->sum);
		}
	}
	//__syncthreads();
}

__global__ void dot_product_BackwardPropogation(neuron* n)
{
	int id = threadIdx.x;
	neuron* self = &n[blockIdx.x];
	//int base = (int)ceilf(log2f(self->in_no));
	//printf("->%d", base);
	__shared__ double d[64];
	if (id < self->out_no)
		d[id] = self->output_n[id]->error * self->output_weight[id];
	else d[id] = 0;
	__syncthreads();
	if (id == 0)
	{
		for (int i = 1; i < 64; i++)
		{
			d[0] += d[i];
		}
		self->error = d[0];
		self->error *= (1 - self->output*self->output);
	}
	//__syncthreads();
}

__global__ void ForwardPropogation(neuron* n)
{
	int id = blockIdx.x + threadIdx.x;
	neuron* self = &n[id];

	//Calculate Sum, Output=>
	self->sum = 0;
	
	for (int i = 0; i < self->in_no; i++)
	{
		self->sum += self->input_weight[i] * self->input_n[i]->output;
	}//*/
	//printf("[%f] ", self->sum);
	if (self->in_no > 0)
	{
		self->output = tanh(self->sum);
	}
}


__global__ void BackPropogation(neuron* n)
{
	int id = blockIdx.x + threadIdx.x;
	neuron* self = &n[id];
	self->error = 0;
	for (int i = 0; i < self->out_no; i++)
	{
		self->error += self->output_n[i]->error * self->output_weight[i];
	}
	self->error *= (1 - self->output*self->output);	// Function Derivative, Derivative of tanh
													//printf("[%f] ", self->error);
}

__global__ void DeltaWeightPropogation(neuron* n)
{
	int id = blockIdx.x + threadIdx.x;
	neuron* self = &n[id];
	for (int i = 0; i < self->in_no; i++)
	{
		self->input_weight[i] -= self->learning_rate * self->input_n[i]->output * self->error;
	}
}

__global__ void linker_thread(neuron* n, int n_no, neuron* in, int in_no)
{
	for (int i = 0; i < in_no; i++)
	{
		in[i].output_weight = new double[n_no];
		in[i].output_n = new neuron*[n_no];
	}
	minstd_rand rng;
	normal_distribution<double> dist(0, 1 / powf(in_no, 0.5));
	for (int i = 0; i < n_no; i++)
	{
		n[i].input_weight = new double[in_no];
		n[i].input_n = new neuron*[in_no];
		for (int j = 0; j < in_no; j++)
		{
			n[i].input_weight[n[i].in_no] = dist(rng);
			n[i].input_n[n[i].in_no] = &in[j];

			in[j].output_weight[in[j].out_no] = dist(rng);
			in[j].output_n[in[j].out_no] = &n[i];

			++n[i].in_no;
			++in[j].out_no;
		}
		printf(">-%d-<", n[i].in_no);
	}
	printf("\nLayer %d Linked to %d and initialized", n_no, in_no);
}

__global__ void initializer_thread(neuron* n, int no)
{
	neural_layer* nl = new neural_layer;
	for (int i = 0; i < no; i++)
	{
		n[i].nl = nl;
	}
	printf("\nLayer %d initialized", no);
}

 void trainer_thread(neuron* in_l, neuron** h_l, int hidden_layers, int* hidden_n, neuron* o_l, idx_content_img* img_train, uint8_t* lbl_train, double* softmax_sum, int _i, int _n)
{
	for (int i = _i; i < _n; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			//Input_Layer_thread << <1, 1 >> > (in_l, img_train, i);
			Input_Layer_Thread << <28 * 28, 1 >> > (in_l, img_train, i);

			for(int m = 0; m < hidden_layers; m++)
				ForwardPropogation << <hidden_n[m], 1 >> > (h_l[m]);

			Output_Layer_SumGenerator_Thread << <10, 1 >> > (o_l);
			Output_Layer_SoftmaxSummation_thread << <1, 1 >> > (o_l, softmax_sum);
			Output_Layer_Thread << <10, 1 >> > (o_l, lbl_train, i, softmax_sum);

			for (int m = 0; m < hidden_layers; m++)
				BackPropogation << <hidden_n[m], 1 >> > (h_l[m]);

			for (int m = 0; m < hidden_layers; m++)
				DeltaWeightPropogation << <hidden_n[m], 1 >> > (h_l[m]);
			DeltaWeightPropogation << <10, 1 >> > (o_l);
			//printf("\nas %d", i);
		}
		//printf("\n->");
	}
	//printf("\nasdssdd");
}
