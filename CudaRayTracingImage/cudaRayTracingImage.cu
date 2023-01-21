//CUDA by example(pdf), page 83
/////////////////////////////
//本例加入了Opengl显示图片
//本例展示了常量内存的用法:存球的信息(可以提升效率)
//简易光线追踪算法：对于每个像素运行一次kernel函数(总共DIM*DIM个线程)
//假设从图像像素[ox,oy]处射出一条射线,离球心越近越亮(fscale越靠近中间越大)
//展示了rnd函数的运用
/////////////////////////////



#include <iostream>
#include <assert.h>
#include "cpu_bitmap.h"

using namespace std;

//#define N 10000000
//#define MAX_ERR 1e-6
//#define imin(a,b) (a<b?a:b)
//const int N = 33 * 1024;
//const int threadsPerBlock = 256; //this is blockDim.x
//const int blocksPerGrid = imin(32, (N+threadsPerBlock-1)/threadsPerBlock);//this is gridDim.x

#define DIM 1024
#define SPHERES 100
#define INF 2e10f
#define rnd(x) (x * rand() / RAND_MAX)

struct Sphere{
	float r,g,b;
	float radius;
	float x,y,z;
	//light is at pixel: [ox,oy]
	__device__ float hit(float ox, float oy, float *fscale){
		float dx = ox - x;
		float dy = oy - y;
		if(dx*dx + dy*dy < radius * radius){
			float dz = sqrtf(radius * radius - dx*dx - dy*dy);
			*fscale = dz / sqrt(radius * radius);
			return dz+z;//这个值只要不是-INF，就属于命中了
		}
		return -INF;
	}
};

__constant__ Sphere s[SPHERES];

__global__ void kernel(unsigned char *ptr) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = x + y * blockDim.x * gridDim.x;
	float ox = x - DIM/2;
	float oy = y - DIM/2;
	
	float r=0, g=0, b=0;
	float maxz = -INF;
	for(int i = 0; i < SPHERES; i++){
		float fscale;
		float t = s[i].hit(ox, oy, &fscale);
		if(t > maxz){
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;
		}
	}

	ptr[offset*4 + 0] = (int)(r * 255);
	ptr[offset*4 + 1] = (int)(g * 255);
	ptr[offset*4 + 2] = (int)(b * 255);
	ptr[offset*4 + 3] = 255;
}

void InitOpenGL_Test(int argc, char **argv){
    glutInit(&argc, argv);
    glutCreateWindow("GLUT"); //must create window to have opengl context
    //glewInit();
	
	char *GL_version=(char *)glGetString(GL_VERSION);
	printf("GL_version: %s\n", GL_version);
    char *GL_vendor=(char *)glGetString(GL_VENDOR);
	printf("GL_vendor: %s\n", GL_vendor);
    char *GL_renderer=(char *)glGetString(GL_RENDERER);
	printf("GL_renderer: %s\n", GL_renderer);
}

int main(int argc, char **argv){
	//InitOpenGL_Test(argc, argv);
	
	//Initialize Host Memory
	CPUBitmap bitmap(DIM,DIM);
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere) * SPHERES);
	for(int i = 0; i < SPHERES; i++){
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;	
		temp_s[i].radius = rnd(100.0f) + 20.0f;	
	}	
	
	//Initialize Device Memory
	unsigned char *d_bitmap;
	cudaMalloc((void**)&d_bitmap, bitmap.image_size());
	cudaMemcpyToSymbol(s, temp_s, sizeof(Sphere) * SPHERES);
	
	//Executing kernel 
	dim3 grids(DIM/16, DIM/16);
	dim3 threads(16, 16);
	kernel<<<grids, threads>>>(d_bitmap);
	
	//Transfer data back to host memory
	cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	
	bitmap.display_and_exit();
	
	//Deallocate device memory
    cudaFree(d_bitmap);
	cudaFree(s);
	//Deallocate host memory
	free(temp_s);
}
