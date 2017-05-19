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

/*__device__ double d_sum_16[16];
__device__ double d_sum_32[32];
__device__ double d_sum_64[64];
__device__ double d_sum_128[128];
__device__ double d_sum_256[256];
__device__ double d_sum_512[512];
__device__ double d_sum_1024[1024];
__device__ double d_sum_2048[2048];
__device__ double d_sum_4096[4096];
__device__ double d_sum_8192[8192];

__global__ void dot_adder_16(int _b)
{
	d_sum_16[threadIdx.x] += d_sum_16[threadIdx.x + _b];
}

__global__ void dot_adder_32(int _b)
{
	d_sum_32[threadIdx.x] += d_sum_32[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_64[threadIdx.x] += d_sum_64[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_64[threadIdx.x] += d_sum_64[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_64[threadIdx.x] += d_sum_64[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_64[threadIdx.x] += d_sum_64[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_64[threadIdx.x] += d_sum_64[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_16[threadIdx.x] += d_sum_16[threadIdx.x + _b];
}

__global__ void dot_adder_16(int _b)
{
	d_sum_16[threadIdx.x] += d_sum_16[threadIdx.x + _b];
}*/

class Dot_Adder
{
public:
	double *dot_sum;

	CUDA_CALLABLE_MEMBER void add(int _b)
	{
		dot_sum[threadIdx.x] += dot_sum[threadIdx.x + _b];
	}
	
	CUDA_CALLABLE_MEMBER Dot_Adder()
	{

	}

	CUDA_CALLABLE_MEMBER void init(int base)
	{
		dot_sum = new double[(int)exp2f(base)];
	}
};

__device__ Dot_Adder dot_adders[12];

__global__ void Dot_Adder_Initialize()
{
	for (int i = 1; i < 12; i++)
	{
		dot_adders[i-1].init(i);
	}
}

__global__ void dot_adder(int base, int _b)
{
	dot_adders[base - 1].add(_b);
}

__global__ void dot_adder_C(Dot_Adder* d_a, int _b)
{
	d_a->add(_b);
}
