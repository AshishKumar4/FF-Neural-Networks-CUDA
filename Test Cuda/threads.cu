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
	//printf("Ax");
	int id = blockIdx.x + threadIdx.x;
	double d = 35;
	n[id].output = tanh((double)(((double)im[index].values[id] - d)) / (d*3.7));
	//printf("Bx ");
}

__global__ void Input_Layer_Thread_vec(neuron* n, int* in_vec, int in_sz, int index)
{
	//printf("Ax");
	int id = blockIdx.x + threadIdx.x;
	n[id].output = ((in_vec[(id*(int)powf(2, in_sz)) + index]));
	//printf("<%f,%d>", n[id].output, (id*(int)powf(2, in_sz)) + index);
}

__global__ void Output_Layer_SumGenerator_Thread(neuron* n)
{
	int id = blockIdx.x + threadIdx.x;
	n[id].sum = -n[id].bias;
	for (int j = 0; j < n[id].in_no; j++)
	{
		n[id].sum += n[id].input_n[j]->output * n[id].input_weight[j];
	}
}

__global__ void Output_Layer_SoftmaxSum_Thread(neuron* n, int out_sz, int o_l_sz_r2, double* tmp)
{
	__shared__ double d_sum[2048];
	//printf("OOO");
	if (threadIdx.x < out_sz)
		d_sum[threadIdx.x] = powf(12, n[threadIdx.x].sum);
	else d_sum[threadIdx.x] = 0;
	//printf("__");
	//__syncthreads();
	//printf("ttA");
	if (threadIdx.x == 0)
	{
		/*int tm = 16;
		for (int i = 0; i < 5; i++)
		{
			tm /= 2;
			dot_adder << <1, tm >> > (4, tm);	// lg16 = 4
												//__syncthreads();
		}*/
		for (int i = 1; i < o_l_sz_r2; i++)
		{
			d_sum[0] += d_sum[i];
		}
		*tmp = d_sum[0];
	}
	//printf("BCCD ");
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

__global__ void Output_Layer_Thread_vec(neuron* n, int* out_vec, int out_sz, int index, double* tmp)
{ 
	double d = *tmp;
	int id = blockIdx.x + threadIdx.x;
	n[id].output = tanhf(n[id].sum);//(tanhf(n[id].sum) + 1) / 2;//powf(12, n[id].sum) / d;
	if (out_vec[(id*(int)powf(2, out_sz)) + index] == 1)
	{
		n[id].error = n[id].output - 1;// -(double)out_vec[(id*(int)powf(2, out_sz)) + index];
	}
	else
	{
		n[id].error = n[id].output;
	}
	n[id].error *= (1 - n[id].output*n[id].output);//n[id].output*(1 - n[id].output);
	printf(" {%f:%d:%f}", n[id].output, out_vec[(id*(int)powf(2, out_sz)) + index], n[id].error);
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
		n[i].sum = -n[i].bias;
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

__global__ void Output_Layer_Test_Thread_vec(neuron* n, int* out_vec, int out_sz, int index, int* output)
{
	int _i = 0;
	double opp = 0;
	/*Softmax Output*/
	double d = 0;
	for (int i = 0; i < out_sz; i++)
	{
		n[i].sum = -n[i].bias;
		for (int j = 0; j < n[i].in_no; j++)
		{
			n[i].sum += n[i].input_n[j]->output * n[i].input_weight[j];
		}
		d += powf(12, n[i].sum);
	}
	for (int i = 0; i < out_sz; i++)
	{
		n[i].output = tanhf(n[i].sum);//(tanhf(n[i].sum) + 1) / 2;//powf(12, n[i].sum) / d;
		if (out_vec[(i*(int)powf(2, out_sz)) + index] - n[i].output == 0)
		{
			++_i;
		}
		printf("[%f]<%d> ", n[i].output, out_vec[(i*(int)powf(2, out_sz)) + index]);
	}

	if (_i == out_sz)
	{
		printf("%d=>", _i);
		*output = *output + 1;
	}
	else printf("\n{%d}", _i);
}

__global__ void dot_product_FeedForward(neuron* n, int r_b)
{
	int id = threadIdx.x;
	neuron* self = &n[blockIdx.x];
	//int base = (int)ceilf(log2f(self->in_no));
	//printf("->%d", base);
	__shared__ double d[2048];
	if (id < self->in_no)
		d[id] = self->input_weight[id] * self->input_n[id]->output;
	else d[id] = 0;
	//__syncthreads();
	if (id == 0)
	{
		for (int i = 1; i < r_b; i++)
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
	//printf("Bps");
	int id = blockIdx.x + threadIdx.x;
	neuron* self = &n[id];
	self->error = 0;
	for (int i = 0; i < self->out_no; i++)
	{
		self->error += self->output_n[i]->error * self->output_weight[i];
	}
	self->error *= (1 - self->output*self->output);	// Function Derivative, Derivative of tanh
													//printf("[%f] ", self->error);
	//printf("Bpe ");
}

__global__ void DeltaWeightPropogation(neuron* n)
{
	//printf("Dps");
	int id = blockIdx.x + threadIdx.x;
	neuron* self = &n[id];
	self->bias += self->learning_rate* ((self->error));
	for (int i = 0; i < self->in_no; i++)
	{
		self->input_weight[i] -= self->learning_rate * ((self->input_n[i]->output * self->error) - (self->input_weight[i] * self->regularization));
	}
	//printf("Dpe ");
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

void trainer_thread(neuron* in_l, neuron** h_l, int hidden_layers, int* hidden_n, neuron* o_l, idx_content_img* img_train, uint8_t* lbl_train, double* softmax_sum, int _i, int _n)
{
	for (int i = _i; i < _n; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			//Input_Layer_thread << <1, 1 >> > (in_l, img_train, i);
			Input_Layer_Thread << <28 * 28, 1 >> > (in_l, img_train, i);

			for (int m = 0; m < hidden_layers; m++)
				ForwardPropogation << <hidden_n[m], 1 >> > (h_l[m]);

			Output_Layer_SumGenerator_Thread << <10, 1 >> > (o_l);
			//Output_Layer_SoftmaxSummation_thread << <1, 1 >> > (o_l, softmax_sum);
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
