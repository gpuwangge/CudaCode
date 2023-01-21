//CUDA by example(pdf), page 73
////////////////////////////
//本例展示了向量点积的简易实现
//kernel前半部分展示了(线程间)共享内存的用法；后半部分展示了归约算法
//最后的结果使用CPU算的
////////////////////////////


#include <iostream>
#include <assert.h>

using namespace std;

//#define N 10000000
//#define MAX_ERR 1e-6
#define imin(a,b) (a<b?a:b)
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

int main(){
	//Initialize Host Memory
    float *a, *b, *out, result; 
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * blocksPerGrid);
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
	
	//Initialize Device Memory
	float *d_a, *d_b, *d_out; 
	cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * blocksPerGrid);
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    //Executing kernel 
	//d_out store partial result
    vector_dot<<<blocksPerGrid,threadsPerBlock>>>(d_out, d_a, d_b);
	
	//Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * blocksPerGrid, cudaMemcpyDeviceToHost);
	
	//finish the result in CPU
	result = 0;
	for(int i = 0; i < blocksPerGrid; i++) result += out[i];
	printf("result = %f\n", result);
	
	//Verification
    //for(int i = 0; i < N; i++){
    //    assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    //}
    //printf("out[0] = %f\n", out[0]);
    //printf("PASSED\n");

	//for(int i = 0; i < 10; i++)
	//	cout<<"out is: "<<out[i]<<endl;
	//getchar();
	
	
	
	//Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);
	//Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
