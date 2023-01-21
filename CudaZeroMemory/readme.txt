
How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
run
	nvcc cudaZeroMemory.cu -o cudaZeroMemory

How to evaluate performance:
run 
	cudaZeroMemory
run
	nvprof cudaZeroMemory


