#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust\device_vector.h"
#include "thrust\host_vector.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "main.h"

using namespace std;
using namespace thrust;


class neuron;
class neural_net;

class Axons
{
public:
	neuron* n;
	double weight;
	CUDA_CALLABLE_MEMBER Axons()
	{
		weight = 22;
	}
	CUDA_CALLABLE_MEMBER ~Axons()
	{

	}
};

class neural_layer
{
public:
	int r_done;
	int e_done;
	int w_done;

	CUDA_CALLABLE_MEMBER neural_layer()
	{
		r_done = 0;
		e_done = 0;
		w_done = 0;
	}
	CUDA_CALLABLE_MEMBER ~neural_layer()
	{
		
	}
};

class neuron
{
public:
	int tmp;
	int r_done;
	int e_done;
	int w_done;

	double output;
	double sum;
	double error;

	double* input_weight;
	neuron** input_n;
	double* output_weight;
	neuron** output_n;

	int in_no;
	int out_no;

	neural_layer* nl;

	double learning_rate;

	CUDA_CALLABLE_MEMBER void link(neuron* n, double weight)
	{
		input_weight[in_no] = 1;
		input_n[in_no] = n;

		n->output_weight[n->out_no] = weight;
		n->output_n[n->out_no] = this;
		++n->out_no;
	}

	CUDA_CALLABLE_MEMBER neuron()
	{
		in_no = 0;
		out_no = 0;
		r_done = 0;
		w_done = 0;
		e_done = 0;
		tmp = 0;
		learning_rate = 0.005;
	}

	CUDA_CALLABLE_MEMBER void init_in(int in)
	{
		input_weight = new double[in];
		input_n = new neuron*[in];
	}

	CUDA_CALLABLE_MEMBER void init_out(int out)
	{
		output_weight = new double[out];
		output_n = new neuron*[out];
	}

	CUDA_CALLABLE_MEMBER ~neuron()
	{

	}
};

class neural_net
{
public:
	int *layer_data;
};