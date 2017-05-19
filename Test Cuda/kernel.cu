
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

#include "threads.h"
#include "dot_adder.h"

using namespace std;
using namespace thrust;

void HighToLowEndian(uint32_t &d)
{
	uint32_t a;
	unsigned char *dst = (unsigned char *)&a;
	unsigned char *src = (unsigned char *)&d;

	dst[0] = src[3];
	dst[1] = src[2];
	dst[2] = src[1];
	dst[3] = src[0];

	d = a;
}

double _gamma = 3.7;

idx_img* imgs, *imgs2;
idx_labels* lbl, *lbl2;

double Func(double x)
{
	return 1;
}


int main()
{
	//cudaSetDevice(0);

	lbl = new idx_labels("digits/trainlabel.bin");
	imgs = new idx_img("digits/trainimg.bin", 60000);

	lbl2 = new idx_labels("digits/testlabel.bin");
	imgs2 = new idx_img("digits/testimg.bin", 10000);

	int hidden_layers = 1;
	int* hidden_n = new int[hidden_layers]{ 128};
	int* nn_n = new int[hidden_layers + 2]{ 768, 128, 10 };

	neuron input_layer[28 * 28];
	neuron** hidden_l = new neuron*[hidden_layers];
	neuron output_layer[10];

	neuron* in_l;
	neuron** h_l = new neuron*[hidden_layers];
	neuron* o_l;


	cudaMalloc(&in_l, sizeof(neuron) * 28 * 28);
	cudaMemcpy(in_l, input_layer, sizeof(neuron) * 28 * 28, cudaMemcpyHostToDevice);

	//neuron** h_l_tmp = new neuron*[hidden_layers];
	neuron** h_l_arr;
	cudaMalloc(&h_l_arr, sizeof(neuron*)*hidden_layers);
	for (int i = 0; i < hidden_layers; i++)
	{
		hidden_l[i] = new neuron[hidden_n[i]];
		cudaMalloc(&h_l[i], sizeof(neuron) * hidden_n[i]);
		cudaMemcpy(h_l[i], hidden_l[i], sizeof(neuron) * hidden_n[i], cudaMemcpyHostToDevice);
		//h_l_tmp[i] = h_l[i];
	}
	cudaMemcpy(h_l_arr, h_l, sizeof(neuron*)*hidden_layers, cudaMemcpyHostToDevice);

	
	cudaMalloc(&o_l, sizeof(neuron) * 10);
	cudaMemcpy(o_l, output_layer, sizeof(neuron) * 10, cudaMemcpyHostToDevice);

	idx_content_img* img_train;
	cudaMalloc(&img_train, sizeof(idx_content_img)*imgs->n_items);
	cudaMemcpy(img_train, imgs->imgs, sizeof(idx_content_img)*imgs->n_items, cudaMemcpyHostToDevice);

	uint8_t* lbl_train;
	cudaMalloc(&lbl_train, sizeof(uint8_t)*lbl->n_items);
	cudaMemcpy(lbl_train, lbl->labels.values, sizeof(uint8_t)*lbl->n_items, cudaMemcpyHostToDevice);

	idx_content_img* img_test;
	cudaMalloc(&img_test, sizeof(idx_content_img)*imgs2->n_items);
	cudaMemcpy(img_test, imgs2->imgs, sizeof(idx_content_img)*imgs2->n_items, cudaMemcpyHostToDevice);

	uint8_t* lbl_test;
	cudaMalloc(&lbl_test, sizeof(uint8_t)*lbl2->n_items);
	cudaMemcpy(lbl_test, lbl2->labels.values, sizeof(uint8_t)*lbl2->n_items, cudaMemcpyHostToDevice);


	int* data;
	cudaMalloc(&data, sizeof(int));

	double* softmax_sum;
	cudaMalloc(&softmax_sum, sizeof(double));

	double tm = 0;
	cudaMemcpy(data, &tm, sizeof(double), cudaMemcpyHostToDevice);

	int* h_n;
	cudaMalloc(&h_n, sizeof(int));
	cudaMemcpy(h_n, hidden_n, sizeof(int)*hidden_layers, cudaMemcpyHostToDevice);

	linker_thread << <1, 1 >> > (h_l[0], hidden_n[0], in_l, 28 * 28);
	cudaDeviceSynchronize();
	int _n_layer = 0;
	for (; _n_layer < hidden_layers - 1; _n_layer++)
	{
		linker_thread << <1, 1 >> > (h_l[_n_layer + 1], hidden_n[_n_layer + 1], h_l[_n_layer], hidden_n[_n_layer]);
		cudaDeviceSynchronize();
	}
	linker_thread << <1, 1 >> > (o_l, 10, h_l[_n_layer], hidden_n[_n_layer]);
	cudaDeviceSynchronize();

	Dot_Adder_Initialize << <1, 1 >> > ();
	cudaDeviceSynchronize();

	printf("\nTraining Begins...");

	int* _out = new int[1];

	for (int k = 0; k < 20; k++)
	{
		for (int i = 0; i < 60000; i++)
		{
			for (int j = 0; j < 1; j++)
			{
				Input_Layer_Thread << <28 * 28, 1 >> > (in_l, img_train, i);

				for (int m = 0; m < hidden_layers; m++)
				{
					//dot_product_FeedForward << <hidden_n[m], 1024 >> > (h_l[m]);
					//cudaDeviceSynchronize();
					ForwardPropogation << <hidden_n[m], 1 >> > (h_l[m]);
				}

				Output_Layer_SumGenerator_Thread << <10, 1 >> > (o_l);
				//Output_Layer_SoftmaxSummation_thread << <1, 1 >> > (o_l, softmax_sum);
				Output_Layer_SoftmaxSum_Thread << <1, 16 >> > (o_l, 10, softmax_sum);
				Output_Layer_Thread << <10, 1 >> > (o_l, lbl_train, i, softmax_sum);

				for (int m = 0; m < hidden_layers; m++)
					BackPropogation << <hidden_n[m], 1 >> > (h_l[m]);
					//dot_product_BackwardPropogation << <hidden_n[m], 64 >> > (h_l[m]);

				for (int m = 0; m < hidden_layers; m++)
					DeltaWeightPropogation << <hidden_n[m], 1 >> > (h_l[m]);
				DeltaWeightPropogation << <10, 1 >> > (o_l);
			}
		}//*/
		cudaDeviceSynchronize();
		cout << "\nEpoch " << k << " Completed";
	}
	printf("\nASDSADASDASD");
	int c = 0;

	*_out = 0;
	cudaMemcpy(data, _out, sizeof(int), cudaMemcpyHostToDevice);
	
	for (int i = 0; i < 10000; i++)
	{
		for (int j = 0; j < 1; j++)
		{
			Input_Layer_Thread << <28*28, 1 >> > (in_l, img_test, i);
			//cudaDeviceSynchronize();
			for (int m = 0; m < hidden_layers; m++)
			{
				ForwardPropogation << <hidden_n[m], 1 >> > (h_l[m]);
				//cudaDeviceSynchronize();
			}
			Output_Layer_Test_Thread << <1, 1 >> > (o_l, lbl_test, i, data);
		}
	}
	//*/
	cudaMemcpy(_out, data, sizeof(int), cudaMemcpyDeviceToHost);
	printf("\n==> %d", *_out);

	cout << "==>" << c;

	//cudaDeviceSynchronize();

	cudaFree(in_l);
	cudaFree(h_l);
	cudaFree(o_l);
	cudaFree(img_train);
	cudaFree(lbl_train);
	int n;
	getchar();
	cin >> n;
	return 0;
}