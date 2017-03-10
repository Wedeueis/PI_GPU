#include <iostream>
#include <string>
#include <sstream>
#include <cuda.h>
#include "ppm.h"
#include "labeling_cuda.cu"


#define START_TIME cudaEventRecord(start,0)
#define STOP_TIME  cudaEventRecord(stop,0 ); \
                   cudaEventSynchronize(stop); \
                   cudaEventElapsedTime( &et, start, stop )


int main(int argc, char* argv[]) {
	cudaEvent_t start,stop;
	cudaEventCreate( &start );
	cudaEventCreate( &stop );
	float et;
	int *label;

	for(int i=0; i<9; i++){
		std::stringstream fname;
		fname << "imagens/teste" << i << ".ppm";
		ppm image(fname.str());

		int w = image.width;
		int h = image.height;

		label = (int*)malloc(w*h*sizeof(int));

		START_TIME;
		CCL(image.r.data(), w, h, label);
		STOP_TIME;

		std::cout << et << "  " ;

		free(label);
	}
	std::cout << std::endl;

	return 0;
}
