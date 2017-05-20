#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

// Round up to the nearest multiple of n
#define ROUNDUP(a, n)						\
({								\
	uint64_t __n = (uint64_t) (n);				\
	(typeof(a)) (ROUNDDOWN((uint64_t) (a) + __n - 1, __n));	\
})