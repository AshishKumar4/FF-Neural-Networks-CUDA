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
#include "main.h"

using namespace std;
using namespace thrust;

class neuron
{
public:
	int tmp;
	int regularization;

	double output;
	double sum;
	double error;
	double bias;

	double* input_weight;
	neuron** input_n;
	double* output_weight;
	neuron** output_n;

	int in_no;
	int out_no;

	double learning_rate;

	CUDA_CALLABLE_MEMBER void link(neuron* n, double weight);

	CUDA_CALLABLE_MEMBER neuron();
	CUDA_CALLABLE_MEMBER neuron(int _regularization);

	CUDA_CALLABLE_MEMBER void init_in(int in);

	CUDA_CALLABLE_MEMBER void init_out(int out);

	CUDA_CALLABLE_MEMBER ~neuron();
};

class NeuralNet_FF
{
public:
	int numLayers;
	vector<neuron*>	Layers;
	vector<int> layerSz;
	vector<int> layerSz_2r;

	int epoch;

	NeuralNet_FF(int n, int* layout, bool auto_link);

	void learn();

	void ForwardProp();

	void Linker(int preL, int postL);
};

class NeuralProcessor
{
public:
	int epoch;
	bool trainer;
	int sampleSize;

	idx_content_img* img;
	uint8_t* lbl;
	int* data;
	int* _out;
	double* softmax_sum;

	NeuralNet_FF* nn;

	NeuralProcessor(NeuralNet_FF* _nn, idx_img* _img, idx_labels* _lbl, int _epoch, int data_sz, bool is_trainer);

	void Run();

	int Results();
};

class NeuralProcessor_MNIST : public NeuralProcessor 
{

};

class NeuralProcessor_vec
{
public:
	int epoch;
	bool trainer;

	int* in_vec;
	int* out_vec;

	int in_sz;
	int out_sz;

	int* data;
	int* _out;
	double* softmax_sum;

	NeuralNet_FF* nn;

	NeuralProcessor_vec(NeuralNet_FF* _nn, vector<int> _in_vec, vector<int> _out_vec, int _in_sz, int _out_sz, int _epoch, bool is_trainer);

	void Run();

	int Results();
};