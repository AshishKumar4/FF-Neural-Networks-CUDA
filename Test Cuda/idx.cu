#pragma once

#include "fstream"
#include "iostream"
#include "string.h"
#include "fstream"

#include "main.h"
#include "idx.h"

using namespace std;

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

CUDA_CALLABLE_MEMBER idx_content::idx_content()
{

}
CUDA_CALLABLE_MEMBER idx_content::~idx_content()
{

}

CUDA_CALLABLE_MEMBER idx_content_img::idx_content_img()
{

}
CUDA_CALLABLE_MEMBER idx_content_img::~idx_content_img()
{

}

idx_file::idx_file(char* fname)
{
	fstream f;
	file = new fstream(fname, ios::binary | ios::in);
	if (file->is_open()) cout << "File Read" << endl;
	else cout << "File Not READ" << endl;

	file->seekg(0);

	file->read((char*)&magic, 4);

	HighToLowEndian(magic);

	file->read((char*)&n_items, 4);

	HighToLowEndian(n_items);

	cout << magic << endl << n_items;
}

idx_labels::idx_labels(char* fname) : idx_file(fname)
{
	labels.values = new uint8_t[n_items];

	file->read((char*)labels.values, n_items);
	cout << "\t " << fname << " File readed successfully. Number of labels: " << n_items << "\n";
}

idx_img::idx_img(char* fname, int n) : idx_file(fname)
{
	imgs = new idx_content_img[n_items];

	file->read((char*)&rows, 4);
	HighToLowEndian(rows);
	file->read((char*)&columns, 4);
	HighToLowEndian(columns);

	int n_size = rows*columns;
	file->seekg(16);

	for (int i = 0; i < n; i++)
	{
		//imgs[i].values = new uint8_t[n_size];
		file->read((char*)imgs[i].values, n_size);
	}
	n_loaded = n;
	cout << "\t " << fname << " File readed successfully. Number of images: " << n_items << ", Loaded images: " << n << "\n";
	cout << rows << " " << columns << endl;
}

idx_img::idx_img(char* fname) : idx_file(fname)
{
	idx_img(fname, n_items);
}