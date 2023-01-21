//CUDA by example(pdf), page 110
/////////////////////////////
//本例展示了热传导的用法(使用纹理内存)
//本例使用texture<float, 2>来替换原先的float *作为device上的内存变量
//!本例子编译未能通过。01/20/2023
/////////////////////////////



#include <iostream>
#include <assert.h>
#include "cpu_bitmap.h"
#include "cpu_anim.h"
#include "book.h"

using namespace std;

//Heat Transfer - 2D Texture memory GPU version

#define SPEED 0.25f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define PI 3.1415926535
#define DIMX 1024//1600
#define DIMY 1024//920

struct DataBlock{
	unsigned char *output_bitmap;
	float *dev_inSrc;
	float *dev_outSrc;
	float *dev_constSrc;
	CPUAnimBitmap *bitmap;
	cudaEvent_t start, stop;
	float totalTime;
	float frames;
};

texture<float, 2> texConstSrc;
texture<float, 2> texIn;
texture<float, 2> texOut;


__global__ void blend_kernel( float *dst, bool dstOut){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float t, l, c, r, b;
	if(dstOut){
		t = tex2D(texIn, x, y-1);
		l = tex2D(texIn, x-1, y);
		c = tex2D(texIn, x, y);
		r = tex2D(texIn, x+1, y);
		b = tex2D(texIn, x, y+1);
	}else{
		t = tex2D(texOut, x, y-1);
		l = tex2D(texOut, x-1, y);
		c = tex2D(texOut, x, y);
		r = tex2D(texOut, x+1, y);
		b = tex2D(texOut, x, y+1);
	}
	dst[offset] = c + SPEED * (t + b + r + l - 4 * c);
}

__global__ void copy_const_kernel(float *iptr){
/*__global__ functions serve as the point of entry into a kernel which executes in parallel on a GPU device.*/
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	float c = tex2D(texConstSrc, x, y);
	if(c != 0)
		iptr[offset] = c;
}


void anim_gpu_TextureMemory(DataBlock *d, int ticks){
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIMX/16, DIMY/16);
	dim3 threads(16, 16);
	CPUAnimBitmap *bitmap = d->bitmap;


	volatile bool dstOut = true;//volatile means stick to program, no optimization
	for(int i=0; i<90; i++){
		float *in, *out;
		if(dstOut){
			in = d->dev_inSrc;
			out = d->dev_outSrc;
		}else{
			out = d->dev_inSrc;
			in = d->dev_outSrc;
		}
		copy_const_kernel<<<blocks, threads>>>(in);
		blend_kernel<<<blocks, threads>>>(out, dstOut);
		dstOut = !dstOut;
	}
	/*for(int i = 0; i < 90; i++){
		copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
		blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
		swap0(d->dev_inSrc, d->dev_outSrc);
	}*/
	float_to_color<<<blocks, threads>>>(d->output_bitmap, d->dev_inSrc);

	cudaMemcpy(bitmap->get_ptr(),
		d->output_bitmap,
		bitmap->image_size(),
		cudaMemcpyDeviceToHost);

	cudaEventRecord(d->stop, 0);
	cudaEventSynchronize(d->stop);

	float elapseTime;
	cudaEventElapsedTime(&elapseTime, d->start, d->stop);

	d->totalTime += elapseTime;

	++d->frames;

	cout<<"Average Time per frame:"<<d->totalTime/d->frames<<endl;
}

void anim_exit_TextureMemory(DataBlock *d){
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);

	cudaUnbindTexture(texIn);
	cudaUnbindTexture(texOut);
	cudaUnbindTexture(texConstSrc);
}

void HeatTransfer_TextureMemory(){//dimx and dimy should be 2^n, 2^m
	/*Create data and bitmap*/
	DataBlock data;
	CPUAnimBitmap bitmap(DIMX, DIMY, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;

	/*Define variable*/
	int imageSize = bitmap.image_size();

	/*Timing*/
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	/*malloc data*/
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());

	cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

	/*Bind Texture*/
	cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
	cudaBindTexture2D(NULL, texConstSrc, data.dev_constSrc, desc, DIMX, DIMY, sizeof(float) * DIMX );
	cudaBindTexture2D(NULL, texIn, data.dev_inSrc, desc, DIMX, DIMY, sizeof(float) * DIMX );
	cudaBindTexture2D(NULL, texOut, data.dev_outSrc, desc, DIMX, DIMY, sizeof(float) * DIMX );

	/*construct temp for data*/
	float *temp = (float*)malloc(bitmap.image_size());
	for(int i = 0; i < DIMX * DIMY; i++){
		temp[i] = 0;
		int x = i % DIMX;
		int y = i / DIMY;
		if((x>300) && (x<600) && (y>310) && (y<601))
			temp[i] = MAX_TEMP; //middle squre const
	}
	temp[DIMY*100 + 100] = ( MAX_TEMP + MIN_TEMP ) / 2; //point const
	temp[DIMY*700 + 100] = MIN_TEMP;
	temp[DIMY*300 + 300] = MIN_TEMP;
	temp[DIMY*200 + 700] = MIN_TEMP;
	for(int y = 800; y < 900; y++){
		for(int x = 400; x < 500; x++){
			temp[x + y*DIMY] = MIN_TEMP;//black hole up
		}
	}
	cudaMemcpy(data.dev_constSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);
	for(int y = 800; y < DIMY; y++){
		for(int x = 0; x < 200; x++){
			temp[x + y*DIMY] = MAX_TEMP;//up left squre input
		}
	}
	cudaMemcpy(data.dev_inSrc, temp, bitmap.image_size(), cudaMemcpyHostToDevice);

	free(temp);

	bitmap.anim_and_exit((void (*)(void*,int))anim_gpu_TextureMemory,// how to use data?
		(void(*)(void*))anim_exit_TextureMemory );
}
