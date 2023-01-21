//CUDA by example(pdf), page 139
////////////////////////////
//本例展示了atomicAdd()的用法。
//kernel里定义了一个share memory temp
//初始化temp, __syncthreads()
//thread可以用atomicAdd()任意写入这一个temp, __syncthreads
//最后把temp的内容写到输出mem里面
////////////////////////////


#include <iostream>
#include <assert.h>
#include "book.h"

using namespace std;
#define SIZE (100*1024*1024)

void Histogram_noGPU(){
	clock_t start = clock();
	unsigned char *buffer = (unsigned char*) big_random_block(SIZE);
	unsigned int histo[256];
	for(int i = 0; i < 256; i++){
		histo[i] = 0;
	}
	for(int i = 0; i < SIZE; i++){
		histo[buffer[i]]++;
	}
	long histoCount = 0;
	for(int i = 0;i < 256; i++){
		histoCount += histo[i];
	}
	cout<<"Histogram Sum: "<<histoCount<<endl;

	free(buffer);

	clock_t finish = clock();
	double currentTime = double(finish - start) / CLOCKS_PER_SEC;
	cout<<"It took "<<currentTime<<" Seconds to construct histograms without GPU."<<endl;
}

__global__ void histo_kernel( unsigned char *buffer, long size, unsigned int *histo){
	__shared__ unsigned int temp[256];
	temp[threadIdx.x] = 0;
	__syncthreads();
	
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int offset = blockDim.x * gridDim.x;

	while(i < size){
		atomicAdd(&temp[buffer[i]], 1);
		i += offset;
	}
	__syncthreads();
	atomicAdd(&(histo[threadIdx.x]), temp[threadIdx.x]);
}

void Histogram_GPUAtomic(){
	unsigned char *buffer = (unsigned char*) big_random_block(SIZE);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	unsigned char *dev_buffer;
	unsigned int *dev_histo;

	cudaMalloc((void**)&dev_buffer, SIZE);
	cudaMemcpy(dev_buffer, buffer, SIZE, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&dev_histo, 256 * sizeof(long));
	cudaMemset(dev_histo, 0, 256 * sizeof(int));

	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	int blocks = prop.multiProcessorCount;
	histo_kernel<<<blocks*2, 256>>>(dev_buffer, SIZE, dev_histo);

	unsigned int histo[256];
	cudaMemcpy(histo, dev_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	cout<<"It took "<<elapseTime/1000<<" Seconds to construct histograms with GPU Atomic."<<endl;

	/*Verify*/
	for(int i = 0; i < SIZE; i++){
		histo[buffer[i]]--;
	}
	for(int i = 0; i < 256; i++){
		if(histo[i] != 0)
			printf("Failure at %d!\n", i);
	}

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(dev_histo);
	cudaFree(dev_buffer);
	free(buffer);

	cout<<"End."<<endl;
}

int main(){
	cout<<"Begin test..."<<endl;
	Histogram_noGPU();
	Histogram_GPUAtomic();
	return 0;
}
