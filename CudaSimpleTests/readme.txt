This simple test is to implement vector add.
cudaSimpleTest_multiBlocks.cu: use 256 blocks and many threads.
cudaSimpleTest_multiThreads.cu: use 1 block and 256 threads.
cudaSimpleTest_singleThread.cu: use 1 block and 1 thread.

How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
nvcc cudaSimpleTest_multiBlocks.cu -o cudaSimpleTest_multiBlocks

How to evaluate performance:
run 
	cudaSimpleTest_multiBlocks, it should show PASSED
run
	nvprof cudaSimpleTest_multiBlocks