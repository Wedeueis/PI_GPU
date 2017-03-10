#include <iostream>
#include <cuda.h>
#include <string>
#include <sstream>
#include "ppm.h"
#include "labeling_cuda.cu"

#define START_TIME cudaEventRecord(start,0)
#define STOP_TIME  cudaEventRecord(stop,0 ); \
                   cudaEventSynchronize(stop); \
                   cudaEventElapsedTime( &et, start, stop )

void threshold(ppm &image){
    for(int i=0; i<image.size; i++){
        if(image.r[i] > 240){
            image.r[i] = (unsigned char)255;
        }else{
            image.r[i] = (unsigned char)0;
        }
    }
}

int main(int argc, char* argv[]) {
    cudaEvent_t start,stop;
    cudaEventCreate( &start );
    cudaEventCreate( &stop );
    float et;
    int *label;
 
    std::string fname = std::string("teste5.ppm");
    ppm image(fname);
    threshold(image);
    image.write("tresh.ppm");

    int w = image.width;
    int h = image.height;

    label = (int*)malloc(w*h*sizeof(int));
    
    START_TIME;
    CCL(&image.r[0], w, h, label);
    STOP_TIME;

    int *ptr = label;

    ppm lbl(w, h);
	
    for(int i = 0; i<h; i++){
	for(int j = 0; j<w; j++){
	    std::cout << *ptr << " ";
	    lbl.r[i*w + j] = *ptr;
	    ptr++;
	}
        std::cout << std::endl;
    }

    lbl.write("label_cuda1.ppm");
    std::cout << et << std::endl;

    free(label);

    return 0;
}
