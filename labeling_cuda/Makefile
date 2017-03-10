all: test_cuda

clean: 
	rm -f test_cuda
	rm -f labelcuda.ppm
	rm -f *.o


ppm.o: ppm.cpp
	g++ $< -o $@ -c

test_cuda: label_test.cu ppm.o
	nvcc $< -o $@ ppm.o
