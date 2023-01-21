//https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/

#include <iostream>
#include <assert.h>

using namespace std;

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float *out, float *a, float *b, int n) {
	int index = threadIdx.x;
	int stride = blockDim.x;

    for(int i = index; i < n; i += stride){
        out[i] = a[i] + b[i];
    }
}

int main(){
	//Initialize Host Memory
    float *a, *b, *out; 
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }
	
	//Initialize Device Memory
	float *d_a, *d_b, *d_out; 
	cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&d_out, sizeof(float) * N);
	cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    //Executing kernel 
    vector_add<<<1,256>>>(d_out, d_a, d_b, N);
	
	//Transfer data back to host memory
    cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);
	
	//Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");

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
