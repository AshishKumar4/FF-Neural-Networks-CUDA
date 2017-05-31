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

#include "main.h"
#include "neuron.h"
#include "threads.h"

using namespace std;

CUDA_CALLABLE_MEMBER void neuron::link(neuron* n, double weight)
{
	input_weight[in_no] = 1;
	input_n[in_no] = n;

	n->output_weight[n->out_no] = weight;
	n->output_n[n->out_no] = this;
	++n->out_no;
}

CUDA_CALLABLE_MEMBER neuron::neuron()
{
	in_no = 0;
	out_no = 0;
	regularization = 0;
	tmp = 0;
	learning_rate = 0.005;
	bias = 1;
}

CUDA_CALLABLE_MEMBER neuron::neuron(int _regularization)
{
	in_no = 0;
	out_no = 0;
	regularization = _regularization;
	tmp = 0;
	learning_rate = 0.005;
	bias = 1;
}

CUDA_CALLABLE_MEMBER void neuron::init_in(int in)
{
	input_weight = new double[in];
	input_n = new neuron*[in];
}

CUDA_CALLABLE_MEMBER void neuron::init_out(int out)
{
	output_weight = new double[out];
	output_n = new neuron*[out];
}

CUDA_CALLABLE_MEMBER neuron::~neuron()
{

}

NeuralNet_FF::NeuralNet_FF(int n, int* layout, bool auto_link = true)
{
	numLayers = n;
	for (int i = 0; i < n; i++)
	{
		neuron *tmp = new neuron[layout[i]];
		neuron *t_l;
		cudaMalloc(&t_l, sizeof(neuron) * layout[i]);
		cudaMemcpy(t_l, tmp, sizeof(neuron) * layout[i], cudaMemcpyHostToDevice);
		Layers.push_back(t_l);
		layerSz.push_back(layout[i]);
		layerSz_2r.push_back((int)exp2f(ceilf(log2f(layout[i]))));
		//delete tmp;
	}
	if (auto_link)
	{
		for (int i = 0; i < n - 1; i++)
		{
			Linker(i, i + 1);
		}
	}
}

void NeuralNet_FF::Linker(int preL, int postL)
{
	linker_thread << <1, 1 >> > (Layers[postL], layerSz[postL], Layers[preL], layerSz[preL]);
	cudaDeviceSynchronize();
}

void NeuralNet_FF::learn()
{
	for (int m = 1; m < numLayers - 1; m++)
		BackPropogation << < layerSz[m], 1 >> > (Layers[m]);
	//dot_product_BackwardPropogation << <hidden_n[m], 64 >> > (h_l[m]);

	for (int m = 1; m < numLayers - 1; m++)
		DeltaWeightPropogation << <layerSz[m], 1 >> > (Layers[m]);

	DeltaWeightPropogation << <layerSz[numLayers - 1], 1 >> > (Layers[numLayers - 1]);
}

void NeuralNet_FF::ForwardProp()
{
	for (int m = 1; m < numLayers; m++)
	{
		//dot_product_FeedForward << <layerSz[m], layerSz_2r[m] >> > (Layers[m], layerSz_2r[m]);
		ForwardPropogation << <layerSz[m], 1 >> > (Layers[m]);
	}
}

NeuralProcessor::NeuralProcessor(NeuralNet_FF* _nn, idx_img* _img, idx_labels* _lbl, int _epoch, int data_sz, bool is_trainer = true)
{
	_out = new int;
	*_out = 0;
	cudaMalloc(&data, sizeof(int));
	cudaMemcpy(data, _out, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&softmax_sum, sizeof(double));

	cudaMalloc(&img, sizeof(idx_content_img)*_img->n_items);
	cudaMemcpy(img, _img->imgs, sizeof(idx_content_img)*_img->n_items, cudaMemcpyHostToDevice);
	cudaMalloc(&lbl, sizeof(uint8_t)*_lbl->n_items);
	cudaMemcpy(lbl, _lbl->labels.values, sizeof(uint8_t)*_lbl->n_items, cudaMemcpyHostToDevice);

	sampleSize = data_sz;
	nn = _nn;
	epoch = _epoch;
	trainer = is_trainer;
}

void NeuralProcessor::Run()
{
	neuron* in_l = nn->Layers[0];
	neuron* o_l = nn->Layers[nn->numLayers - 1];
	int o_l_sz = nn->layerSz[nn->numLayers - 1];
	int o_l_sz_r2 = (int)exp2f(ceilf(log2f(o_l_sz)));

	for (int k = 0; k < epoch; k++)
	{
		for (int i = 0; i < sampleSize; i++)
		{
			Input_Layer_Thread << <nn->layerSz[0], 1 >> > (in_l, img, i);

			nn->ForwardProp();

			if (trainer)
			{
				//Output_Layer_SumGenerator_Thread << <o_l_sz, 1 >> > (o_l);
				Output_Layer_SoftmaxSum_Thread << <1, o_l_sz_r2 >> > (o_l, o_l_sz, o_l_sz_r2, softmax_sum);
				Output_Layer_Thread << <o_l_sz, 1 >> > (o_l, lbl, i, softmax_sum);

				nn->learn();
			}
			else
			{
				Output_Layer_Test_Thread << <1, 1 >> > (o_l, lbl, i, data);
			}
		}
		cudaDeviceSynchronize();
		if(trainer)
			cout << "\nEpoch " << k << " Completed";
		else cout << "\nTesting Complete";
	}
}

int NeuralProcessor::Results()
{
	cudaMemcpy(_out, data, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\nResults ==> %d", *_out);
	return *_out;
}

NeuralProcessor_vec::NeuralProcessor_vec(NeuralNet_FF* _nn, vector<int> _in_vec, vector<int> _out_vec, int _in_sz, int _out_sz, int _epoch, bool is_trainer)
{
	_out = new int;
	*_out = 0;
	cudaMalloc(&data, sizeof(int));
	cudaMemcpy(data, _out, sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&softmax_sum, sizeof(double));

	cudaMalloc(&in_vec, sizeof(int)*_in_sz * pow(2, _in_sz));
	cudaMalloc(&out_vec, sizeof(int)*_out_sz * pow(2, _in_sz));

	in_sz = _in_sz;
	out_sz = _out_sz;

	cudaMemcpy(in_vec, (void*)&_in_vec[0], sizeof(int)*_in_vec.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(out_vec, (void*)&_out_vec[0], sizeof(int)*_out_vec.size(), cudaMemcpyHostToDevice);

	nn = _nn;
	epoch = _epoch;
	trainer = is_trainer;
}

void NeuralProcessor_vec::Run()
{
	neuron* in_l = nn->Layers[0];
	neuron* o_l = nn->Layers[nn->numLayers - 1];
	int o_l_sz = nn->layerSz[nn->numLayers - 1];
	int o_l_sz_r2 = (int)exp2f(ceilf(log2f(o_l_sz)));

	for (int i = 0; i < epoch; i++)
	{
		for (int j = 0; j < (int)pow(2, in_sz); j++)
		{
			Input_Layer_Thread_vec << <nn->layerSz[0], 1 >> > (in_l, in_vec, in_sz, j);
			cout << endl;

			nn->ForwardProp();
			if (trainer)
			{
				Output_Layer_SumGenerator_Thread << <o_l_sz, 1 >> > (o_l);
				//Output_Layer_SoftmaxSum_Thread << <1, o_l_sz_r2 >> > (o_l, o_l_sz, o_l_sz_r2, softmax_sum);
				Output_Layer_Thread_vec << <o_l_sz, 1 >> > (o_l, out_vec, out_sz, j, softmax_sum);

				nn->learn();
			}
			else
			{
				Output_Layer_Test_Thread_vec << <1, 1 >> > (o_l, out_vec, out_sz, j, data);
			}
			cout << endl;
		}
		cudaDeviceSynchronize();
		if (trainer)
			cout << "\nEpoch " << i << " Completed";
		else cout << "\nTesting Complete";
	}
}

int NeuralProcessor_vec::Results()
{
	/*int n0, n1, o0;
	cout << "\nEnter input0: ";
	cin >> n0;
	cout << "\nEnter input1: ";
	cin >> n1;
	
	Input_Layer_Thread_vec << <nn->layerSz[0], 1 >> > (in_l, in_vec, in_sz, j);*/
	return 0;
}