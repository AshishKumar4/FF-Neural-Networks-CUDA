#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "dot_adder.h"
#include "main.h"

CUDA_CALLABLE_MEMBER void Dot_Adder::add(int id, int _b)
{
	dot_sum[id] += dot_sum[id + _b];
}

CUDA_CALLABLE_MEMBER Dot_Adder::Dot_Adder()
{

}

CUDA_CALLABLE_MEMBER void Dot_Adder::init(int base)
{
	dot_sum = new double[(int)exp2f(base)];
}

__device__ Dot_Adder dot_adders[12];

__global__ void Dot_Adder_Initialize()
{
	for (int i = 1; i < 12; i++)
	{
		dot_adders[i - 1].init(i);
	}
}

__global__ void dot_adder(int base, int _b)
{
	dot_adders[base - 1].add(threadIdx.x, _b);
}

__global__ void dot_adder_C(Dot_Adder* d_a, int _b)
{
	d_a->add(threadIdx.x, _b);
}