#include "labeling.h"

using namespace std;

int main(){
	string fname = std::string("gears.ppm");
	ppm image(fname);
	int lin = image.height;
	int col = image.width;
	ppm newimage(lin, col);
	int label = -1;
	for(int i = 0; i<lin; i++)
		for(int j =0; j<col; j++) {
			cout << i << " " <<  j << endl;
			labeling(image, newimage, i,j,label);
			label--;
		}
	newimage.write("label.ppm");
	return 0;
}


