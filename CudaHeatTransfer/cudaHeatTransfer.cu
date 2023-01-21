//CUDA by example(pdf), page 100
/////////////////////////////
//本例展示了热传导的用法(未使用纹理内存)
//展示了cpu_anim.h动画效果函数的使用：定义函数：bitmap.anim_and_exit(), 然后传入两个函数anim_gpu() and anim_exit()
//anim_gpu(): 一个长度90的loop, 连续call copy_const_kernel()和blend_kernel()和swap()
//anim_exit(): call cudaFree()
//三个device memcory variable:
//float *dev_constSrc; //存常量热量图
//float *dev_inSrc; //copy_const_kernel()把dev_constSrc拷贝到dev_inSrc
//float *dev_outSrc; //blend_kernel()使用dev_inSrc作为input来更新dev_outSrc, 然后swap两者
//展示了cudaEvent的使用
/////////////////////////////



#include <iostream>
#include <assert.h>
#include "cpu_bitmap.h"
#include "cpu_anim.h"
#include "book.h"

using namespace std;

//Heat Transfer - normal GPU version

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

__global__ void copy_const_kernel(float *iptr, const float *cptr){
/*__global__ functions serve as the point of entry into a kernel which executes in parallel on a GPU device.*/
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	if(cptr[offset] != 0) iptr[offset] = cptr[offset];
}

__global__ void blend_kernel( float *outSrc, const float *inSrc){
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;

	int left = offset - 1;
	int right = offset + 1;
	if(x == 0) left++;
	if(x == DIMX- 1) right--;

	int top = offset - DIMY;
	int bottom = offset + DIMY;
	if(y == 0) top += DIMY;
	if(y == DIMY - 1) bottom -= DIMY;

	outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top] +
		inSrc[bottom] + inSrc[left] + inSrc[right] -
		inSrc[offset]*4);
}

void anim_gpu(DataBlock *d, int ticks){
	cudaEventRecord(d->start, 0);
	dim3 blocks(DIMX/16, DIMY/16);
	dim3 threads(16, 16);
	CPUAnimBitmap *bitmap = d->bitmap;

	for(int i = 0; i < 90; i++){
		copy_const_kernel<<<blocks, threads>>>(d->dev_inSrc, d->dev_constSrc);
		blend_kernel<<<blocks, threads>>>(d->dev_outSrc, d->dev_inSrc);
	//	std::swap(d->dev_inSrc, d->dev_outSrc);//not default swap
		swap0(d->dev_inSrc, d->dev_outSrc);
	}
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

void anim_exit(DataBlock *d){
	cudaFree(d->dev_inSrc);
	cudaFree(d->dev_outSrc);
	cudaFree(d->dev_constSrc);

	cudaEventDestroy(d->start);
	cudaEventDestroy(d->stop);
}

void main(){ // must be 2^n by 2^m
	/*Create data and bitmap*/
	DataBlock data;
	CPUAnimBitmap bitmap(DIMX, DIMY, &data);
	data.bitmap = &bitmap;
	data.totalTime = 0;
	data.frames = 0;

	/*Timing*/
	cudaEventCreate(&data.start);
	cudaEventCreate(&data.stop);

	/*malloc data*/
	cudaMalloc((void**)&data.output_bitmap, bitmap.image_size());

	cudaMalloc((void**)&data.dev_inSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_outSrc, bitmap.image_size());
	cudaMalloc((void**)&data.dev_constSrc, bitmap.image_size());

	/*construct temp for data*/
	float *temp = (float*)malloc(bitmap.image_size());
	for(int i = 0; i < DIMX * DIMY; i++){
		temp[i] = 0;
		int x = i % DIMX;
		int y = i / DIMY;
		if((x>400) && (x<500) && (y>310) && (y<601))
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

	bitmap.anim_and_exit((void (*)(void*,int))anim_gpu,// how to use data?
		(void(*)(void*))anim_exit );

}
