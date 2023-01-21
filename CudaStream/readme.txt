
How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
run
	nvcc cudaStream.cu -o cudaStream

How to evaluate performance:
run 
	cudaStream
run
	nvprof cudaStream