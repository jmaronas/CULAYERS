GPU_CXX := nvcc

all: libculayers.so

cublas_env.o: ./cublas_env.h ./utils_gpu_env.h
	$(GPU_CXX)  --shared --compiler-options '-fPIC' -c ./cublas_env.cu

gpu_env.o: ./gpu_env.h ./gpu_kernels.h ./utils.h  ./utils_gpu_env.h
	$(GPU_CXX) --shared --compiler-options '-fPIC' -c ./gpu_env.cu

gpu_kernels.o:  ./gpu_kernels.h
	$(GPU_CXX) --shared --compiler-options '-fPIC' -c ./gpu_kernels.cu

utils.o:  ./utils.h
	$(GPU_CXX) --shared --compiler-options '-fPIC' -c ./utils.cu

libculayers.so:  utils.o gpu_kernels.o gpu_env.o cublas_env.o
	nvcc --shared --compiler-options '-fPIC' -o libculayers.so -lm -lcudart -lcublas -lcurand utils.o gpu_kernels.o gpu_env.o cublas_env.o 
clean:
	rm *.o

