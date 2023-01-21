//CUDA by example(pdf), page 139
////////////////////////////
//本例展示了Stream,可以使GPU实现多任务
////////////////////////////


#include <iostream>
#include <assert.h>

using namespace std;

#define N2 (1024*1024)
#define FULL_DATA_SIZE (N2*20)
__global__ void kernel_stream(int *a, int *b, int *c){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if(idx < N2){
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}

int main(){
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if(!prop.deviceOverlap){
		cout<<"The divice will not handle overlaps, so no speed up from streams."<<endl;
		return;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/*Init streams*/
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

	/*Create identical stes of GPU buffers*/
	int *dev_a0, *dev_b0, *dev_c0;
	int *dev_a1, *dev_b1, *dev_c1;
	cudaMalloc((void**)&dev_a0, N2 * sizeof(int));
	cudaMalloc((void**)&dev_b0, N2 * sizeof(int));
	cudaMalloc((void**)&dev_c0, N2 * sizeof(int));
	cudaMalloc((void**)&dev_a1, N2 * sizeof(int));
	cudaMalloc((void**)&dev_b1, N2 * sizeof(int));
	cudaMalloc((void**)&dev_c1, N2 * sizeof(int));

	/*Allocate page-locked memory for stream*/
	int *host_a, *host_b, *host_c;
	cudaHostAlloc((void**)&host_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	cudaHostAlloc((void**)&host_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault);
	for(int i = 0; i < FULL_DATA_SIZE; i++){
		host_a[i] = rand();
		host_b[i] = rand();
	}

	
	for(int i = 0; i < FULL_DATA_SIZE; i += N2 * 2){
		///*Stream0*/
		//cudaMemcpyAsync(dev_a0, host_a + i, N2 * sizeof(int), cudaMemcpyHostToDevice, stream0);
		//cudaMemcpyAsync(dev_b0, host_b + i, N2 * sizeof(int), cudaMemcpyHostToDevice, stream0);
		//kernel_stream<<<N2/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		//cudaMemcpyAsync(host_c + i, dev_c0, N2 * sizeof(int), cudaMemcpyDeviceToHost, stream0);

		///*Stream1*/
		//cudaMemcpyAsync(dev_a1, host_a + i + N2, N2 * sizeof(int), cudaMemcpyHostToDevice, stream1);
		//cudaMemcpyAsync(dev_b1, host_b + i + N2, N2 * sizeof(int), cudaMemcpyHostToDevice, stream1);
		//kernel_stream<<<N2/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
		//cudaMemcpyAsync(host_c + i + N2, dev_c1, N2 * sizeof(int), cudaMemcpyDeviceToHost, stream1);

		/*Optimized version*/
		cudaMemcpyAsync(dev_a0, host_a + i, N2 * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_a1, host_a + i + N2, N2 * sizeof(int), cudaMemcpyHostToDevice, stream1);
		cudaMemcpyAsync(dev_b0, host_b + i, N2 * sizeof(int), cudaMemcpyHostToDevice, stream0);
		cudaMemcpyAsync(dev_b1, host_b + i + N2, N2 * sizeof(int), cudaMemcpyHostToDevice, stream1);
		kernel_stream<<<N2/256, 256, 0, stream0>>>(dev_a0, dev_b0, dev_c0);
		kernel_stream<<<N2/256, 256, 0, stream1>>>(dev_a1, dev_b1, dev_c1);
		cudaMemcpyAsync(host_c + i, dev_c0, N2 * sizeof(int), cudaMemcpyDeviceToHost, stream0);
		cudaMemcpyAsync(host_c + i + N2, dev_c1, N2 * sizeof(int), cudaMemcpyDeviceToHost, stream1);
	}

	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	cout<<"It took "<<elapseTime/1000<<" Seconds."<<endl;

	cudaFreeHost(host_a);
	cudaFreeHost(host_b);
	cudaFreeHost(host_c);
	cudaFreeHost(dev_a0);
	cudaFreeHost(dev_b0);
	cudaFreeHost(dev_c0);
	cudaFreeHost(dev_a1);
	cudaFreeHost(dev_b1);
	cudaFreeHost(dev_c1);
	cudaStreamDestroy(stream0);
	cudaStreamDestroy(stream1);
	return 0;
}
