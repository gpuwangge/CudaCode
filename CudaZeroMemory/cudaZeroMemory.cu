//Cuda by example pdf (Page 166)
////////////////////////////
//本例展示了用zero memory来计算向量点积
//Zero Memory, will use dot(float *a, float *b, float *c) in chapter 5, will use N
//本来使用cudaHostAlloc()和cudaHostAllocDefault参数来获得默认的固定内存。
//但如果换成cudaHostAllocMapped参数，可以获得另一种固定内存：这种固定内存甚至不需要复制到GPU上，CUDA C Kernel函数可以直接访问。
//限制：硬件设备必须支持零拷贝内存。
//对于集成GPU，因为内存本来就跟CPU共享，使用零拷贝内存就可以避免不必要的复制，从而提升效率。
//对于独立显卡稍微复杂一点：如果GPU只访问该内存一次，那么零拷贝内存也有提升效率的效果；
//但如果需要多次访问，还不如把内存里的内容拷贝到GPU上
////////////////////////////

#include <iostream>
using namespace std;

#define imin(a,b) (a<b?a:b)
#define sum_squares(x) (x * ( x + 1) * (2 * x + 1) / 6)
const int N = 33 * 1024;
const int threadsPerBlock = 256; //this is blockDim.x
const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);//this is gridDim.x

__global__ void vector_dot(float *out, float *a, float *b) {
	__shared__ float cache[threadsPerBlock];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	//int cacheIndex = threadIdx.x;
	
	float tmp = 0;
	while(tid < N){
		tmp += a[tid] * b[tid];
		tid += threadsPerBlock * blocksPerGrid;//blockDim.x * gridDim.x; same effect
	}
	
	cache[threadIdx.x] = tmp;
	
	__syncthreads();
	
	int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i)
			cache[threadIdx.x] += cache[threadIdx.x + i];
		__syncthreads();
		i /= 2;		
	}
	if(threadIdx.x == 0)
		out[blockIdx.x] = cache[0];
}

 void cuda_host_alloc_test(int size){
	cout<<"Size is "<<size<<"."<<endl;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cout<<"Start timer..."<<endl;
	cudaEventRecord(start, 0);

	float *a, *b, c, *partial_c;
	float *dev_a, *dev_b, *dev_partial_c;

	cudaHostAlloc((void**)&a, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&b, size*sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped);
	cudaHostAlloc((void**)&partial_c, blocksPerGrid*sizeof(float), cudaHostAllocMapped);

	for(int i =0; i < N; i++){
		a[i] = i;
		b[i] = i * 2;
	}

	cudaHostGetDevicePointer(&dev_a, a, 0);
	cudaHostGetDevicePointer(&dev_b, b, 0);
	cudaHostGetDevicePointer(&dev_partial_c, partial_c, 0);
	
	cout<<"Call kernel. ("<<blocksPerGrid<<","<<threadsPerBlock<<")"<<endl;
	vector_dot<<<blocksPerGrid, threadsPerBlock>>>(dev_partial_c, dev_a, dev_b);

	cout<<"Synchronize."<<endl;
	cudaThreadSynchronize();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapseTime;
	cudaEventElapsedTime(&elapseTime, start, stop);
	cout<<"It took "<<elapseTime/1000<<" Seconds."<<endl;

	c = 0;
	for (int i = 0; i < blocksPerGrid; i++){
		c += partial_c[i];
	}

	printf("Does GPU value C = %.6g =? %.6g\n", c,
		2 * sum_squares( (float) (N - 1) ) );

	cudaFreeHost(a);
	cudaFreeHost(b);
	cudaFreeHost(partial_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


 }

void main(){
	cudaDeviceProp prop;
	int whichDevice;
	cudaGetDevice(&whichDevice);
	cudaGetDeviceProperties(&prop, whichDevice);

	if(prop.canMapHostMemory != 1){
		cout<<"The divice cannot map memory."<<endl;
		return;
	}

	cout<<"Set device flag."<<endl;
	cudaSetDeviceFlags(cudaDeviceMapHost);


	cuda_host_alloc_test(N);
}