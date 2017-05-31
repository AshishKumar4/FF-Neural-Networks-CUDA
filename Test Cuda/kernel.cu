
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

void digit_recog()
{
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
}

void logic_emulate()
{
	int epoch;
	cout << "Enter epoch: ";
	cin >> epoch;
	int hn;
	cout << "\nEnter Hidden Neurons: ";
	cin >> hn;
	int n, m;
	cout << "Enter the number of input vectors: ";
	cin >> n;
	cout << "Enter the number of output vectors: ";
	cin >> m;
	vector<int> arg;
	vector<int> out;
	cout << "\nEnter the Input Vectors =>";
	for (int i = 0; i < n; i++)
	{
		cout << "\nEnter the vector " << i << " :";
		int t2;
		for (int j = 0; j < pow(2, n); j++)
		{
			cin >> t2;
			arg.push_back(t2);
		}
	}

	cout << "\nEnter the Output Vectors =>";
	for (int i = 0; i < m; i++)
	{
		cout << "\nEnter the vector " << i << " :";
		int t2;
		for (int j = 0; j < pow(2, n); j++)
		{
			cin >> t2;
			out.push_back(t2);
		}
	}
	cout << "Vector Table input successful, Learning!";

	//Create a neural network
	int* nn_n = new int[3]{ n, hn, m };
	NeuralNet_FF* nn = new NeuralNet_FF(3, nn_n, true);

	NeuralProcessor_vec Trainer(nn, arg, out, n, m, epoch, true);
	Trainer.Run();

	NeuralProcessor_vec Tester(nn, arg, out, n, m, 1, false);
	Tester.Run();
}

int main()
{
	//cudaSetDevice(0);

	digit_recog();
	//logic_emulate();
	int n;
	cin >> n;
	return 0;
}