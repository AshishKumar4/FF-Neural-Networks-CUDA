
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <vector>

#include "idx.h"
#include "neuron.h"

#include "threads.h"
#include "dot_adder.h"

using namespace std;
using namespace thrust;

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

	int* nn_n = new int[3]{ 768, 40, 10 };
	NeuralNet_FF* nn = new NeuralNet_FF(3, nn_n, true);
	NeuralProcessor Trainer(nn, imgs, lbl, 1, 60000, true);
	cout << "\nTraining Starts...";
	Trainer.Run();

	NeuralProcessor Tester(nn, imgs2, lbl2, 1, 10000, false);
	Tester.Run();
	Tester.Results();
	int n;
	cin >> n;
	return 0;
}