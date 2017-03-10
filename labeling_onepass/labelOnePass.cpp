#include "ppm.h"
#include <iostream>
#include <queue>
#include <string>

using namespace std;

struct pos{
	int i;
	int j;
};

bool labeling(ppm &source, int i, int j, int label){
	int index = i*source.width + j;
	unsigned int pixel_value = (unsigned int) source.r[index];
	if(pixel_value != 255)
		return false;
	int lin = source.height;
	int col = source.width;
	queue <pos> positions;
	pos current;
	current.i = i;
	current.j = j; 
	positions.push(current);
	while(!positions.empty() ){
		pos n = positions.front();
		positions.pop();
		vector<pos> v;
		if(n.j+1<col){
			pos viz;
			viz.i = n.i;
			viz.j = n.j + 1;
			v.push_back(viz);
		}
		if(n.j-1>=0){
			pos viz;
			viz.i = n.i;
			viz.j = n.j - 1;
			v.push_back(viz);
		}
		if(n.i+1<lin){
			pos viz;
			viz.i = n.i + 1;
			viz.j = n.j;
			v.push_back(viz);
		}
		if(n.i-1>=0){
			pos viz;
			viz.i = n.i - 1;
			viz.j = n.j;
			v.push_back(viz);
		}
		for(int k = 0; k<v.size(); k++) {
			int idx = v[k].i*source.width + v[k].j;
			if(source.r[idx] == 255){
				source.r[idx] = (unsigned char)label;
				positions.push(v[k]);
			}
		}

	}
	return true;
}

int main(){

	string fname = std::string("gears2.ppm");
	ppm image(fname);

	int lin = image.height;
	int col = image.width;
	int label = 1;
	for(int i = 0; i<lin; i++)
		for(int j =0; j<col; j++) {
			bool contou = labeling(image, i,j,label);
			if(contou) label+=100;
		}
	image.write("labelonepass.ppm");
	return 0;
}

//conversao de indices x, y => k
// k = x + y*width
// x = k%width
// y = (k)/width