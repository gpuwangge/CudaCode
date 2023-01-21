
How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
Install OpenGL

run
	nvcc cudaHeatTransfer.cu -o cudaHeatTransfer

How to evaluate performance:
run 
	cudaHeatTransfer
run
	nvprof cudaHeatTransfer

===============本例编译未能通过===================
	nvcc cudaHeatTransfer_2dTexture.cu -o cudaHeatTransfer_2dTexture

	cudaHeatTransfer_2dTexture

	nvprof cudaHeatTransfer_2dTexture

