#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "main.h"

class Dot_Adder
{
public:
	double *dot_sum;

	CUDA_CALLABLE_MEMBER void add(int id, int _b);
	
	CUDA_CALLABLE_MEMBER Dot_Adder();

	CUDA_CALLABLE_MEMBER void init(int base);
};

extern __device__ Dot_Adder dot_adders[12];

__global__ void Dot_Adder_Initialize();

__global__ void dot_adder(int base, int _b);

__global__ void dot_adder_C(Dot_Adder* d_a, int _b);