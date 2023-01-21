This simple test is to implement vector add.
cudaDotProduct.cu: use many blocks and 256 threads.


How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
run
	nvcc cudaDotProduct.cu -o cudaDotProduct

How to evaluate performance:
run 
	cudaDotProduct
run
	nvprof cudaDotProduct

=======================
	nvcc cudaDotProductAtomic.cu -o cudaDotProductAtomic
	cudaDotProductAtomic
	nvprof cudaDotProductAtomic
