
How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
run
	nvcc cudaAtomicHistogram.cu -o cudaAtomicHistogram

How to evaluate performance:
run 
	cudaAtomicHistogram
run
	nvprof cudaAtomicHistogram