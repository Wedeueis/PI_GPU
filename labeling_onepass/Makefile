all: labelOnePass

clean: 
	rm -f labelOnePass
	rm -f *.o
	rm -f labelonepass.ppm
	rm -f imgcopy.ppm

ppm.o: ppm.cpp
	g++ $< -o $@ -c

labelOnePass: labelOnePass.cpp ppm.o
	g++ $< -o $@ ppm.o
