#pragma once

#include "fstream"
//#include "stdafx.h"
#include "iostream"
#include "string.h"
#include "fstream"

#include "main.h"

using namespace std;

void HighToLowEndian(uint32_t &d);

class idx_content
{
public:
	uint8_t* values;
	CUDA_CALLABLE_MEMBER idx_content();

	CUDA_CALLABLE_MEMBER ~idx_content();
};

class idx_content_img
{
public:
	uint8_t values[28*28];
	CUDA_CALLABLE_MEMBER idx_content_img();

	CUDA_CALLABLE_MEMBER ~idx_content_img();
};

class idx_file
{
protected:
	fstream* file;

	uint32_t magic;

public:
	uint32_t n_items;

	idx_file(char* fname);
};

class idx_labels : public idx_file
{
public:

	idx_content labels;
	idx_labels(char* fname);
};

class idx_img : public idx_file
{
public:
	uint32_t rows;
	uint32_t columns;

	idx_content_img* imgs;
	int n_loaded;

	idx_img(char* fname, int n);

	idx_img(char* fname);
};

