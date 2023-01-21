#include <iostream>

__global__ void mykernel(void) {
}
int main(void) {
	mykernel<<<1,1>>>();
	//printf("Hello World!\n");
	std::cout<<"Cuda Hello World!"<<std::endl;
	return 0;
}
