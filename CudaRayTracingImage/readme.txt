

How to run:
Install Cuda Toolkit.
Add cl.exe to system environment\PATH
Install OpenGL

run
	nvcc cudaRayTracingImage.cu -o cudaRayTracingImage

How to evaluate performance:
run 
	cudaRayTracingImage
run
	nvprof cudaRayTracingImage