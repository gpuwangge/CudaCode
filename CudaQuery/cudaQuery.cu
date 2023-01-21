#include <iostream>
using namespace std;

void main(){
	int count;
	cudaGetDeviceCount( &count );
	cout<<count<<" device(s) is found."<<endl;

	cudaDeviceProp prop;
	for(int i = 0; i < count; i++){
		cudaGetDeviceProperties( &prop, i);
		cout<<"Information of device "<<i + 1<<endl;
		cout<<prop.name<<endl;
		cout<<"Compute capability: "<<prop.major<<"."<<prop.minor<<endl;
		cout<<"Clock rate: "<<prop.clockRate<<endl;
		cout<<"Total global memory: "<<prop.totalGlobalMem<<endl;
		cout<<"Total constant memory: "<<prop.totalConstMem<<endl;
		cout<<"Max memory pitch: "<<prop.memPitch<<endl;
		cout<<"Multiprocessor count: "<<prop.multiProcessorCount<<endl;
		cout<<"Shared memory per block: "<<prop.sharedMemPerBlock<<endl;
		cout<<"Max threads per block: "<<prop.maxThreadsPerBlock<<endl;
		cout<<"Max thread dimensions: "<<prop.maxThreadsDim[0]<<" "<<prop.maxThreadsDim[1]<<" "<<prop.maxThreadsDim[2]<<endl;
		cout<<"Max grid dimensions: "<<prop.maxGridSize[0]<<" "<<prop.maxGridSize[1]<<" "<<prop.maxGridSize[2]<<endl;
	
	}
}
