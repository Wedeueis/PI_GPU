#include "labeling.h"
using namespace std;

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <exception>
#include <vector>

//init with default values

void ppm::init() {
    width = 0;
    height = 0;
    max_col_val = 255;
}

//create a PPM object

ppm::ppm() {
    init();
}

//create a PPM object and fill it with data stored in fname 

ppm::ppm(const std::string &fname) {
    init();
    read(fname);
}

//create an "epmty" PPM image with a given width and height;the R,G,B arrays are filled with zeros

ppm::ppm(const unsigned int _width, const unsigned int _height) {
    init();
    width = _width;
    height = _height;
    nr_lines = height;
    nr_columns = width;
    size = width*height;

    r.resize(size);
    g.resize(size);
    b.resize(size);

    for (unsigned int i = 0; i < size; ++i) {
        r[i] = 0;
        g[i] = 0;
        b[i] = 0;
    }
}

//read the PPM image from fname

void ppm::read(const std::string &fname) {
    std::ifstream inp(fname.c_str(), std::ios::in | std::ios::binary);
    if (inp.is_open()) {
        std::string line;
        std::getline(inp, line);
        if (line != "P6") {
            std::cout << "Error. Unrecognized file format." << std::endl;
            return;
        }
        std::getline(inp, line);
        while (line[0] == '#') {
            std::getline(inp, line);
        }
        std::stringstream dimensions(line);

        try {
            dimensions >> width;
            dimensions >> height;
            nr_lines = height;
            nr_columns = width;
        } catch (std::exception &e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        std::getline(inp, line);
        std::stringstream max_val(line);
        try {
            max_val >> max_col_val;
        } catch (std::exception &e) {
            std::cout << "Header file format error. " << e.what() << std::endl;
            return;
        }

        size = width*height;

        r.reserve(size);
        g.reserve(size);
        b.reserve(size);

        char aux;
        for (unsigned int i = 0; i < size; ++i) {
            inp.read(&aux, 1);
            r[i] = (unsigned char) aux;
            inp.read(&aux, 1);
            g[i] = (unsigned char) aux;
            inp.read(&aux, 1);
            b[i] = (unsigned char) aux;
        }
    } else {
        std::cout << "Error. Unable to open " << fname << std::endl;
    }
    inp.close();
}

//write the PPM image in fname

void ppm::write(const std::string &fname) {
    std::ofstream inp(fname.c_str(), std::ios::out | std::ios::binary);
    if (inp.is_open()) {

        inp << "P6\n";
        inp << width;
        inp << " ";
        inp << height << "\n";
        inp << max_col_val << "\n";

        char aux;
        for (unsigned int i = 0; i < size; ++i) {
            aux = (char) r[i];
            inp.write(&aux, 1);
            aux = (char) g[i];
            inp.write(&aux, 1);
            aux = (char) b[i];
            inp.write(&aux, 1);
        }
    } else {
        std::cout << "Error. Unable to open " << fname << std::endl;
    }
    inp.close();
}

ppm labeling(ppm &source, ppm &newimage, int i, int j, int label) {
	
	int index = i*source.width + j;
	std::cout << source.r.size() << endl;
	if(index == (source.width * source.height) ) index -= 1;
	unsigned int pixel_value = (unsigned int) source.r[index];
	std::cout <<"teste" << endl;
	if(pixel_value != 255)
		return source;
	for(int x = 0; x<source.size; x++)
		newimage.r[x] = source.r[x];
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
				newimage.r[idx] = label;
				positions.push(v[k]);
			}
		}


}

	return newimage;
	
}
