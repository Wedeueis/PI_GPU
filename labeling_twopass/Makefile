all: labelTwoPass

clean:
	rm -f labelTwoPass
	rm -f *.o
	rm -f labeltwopass.ppm
	
ppm.o: ppm.cpp
	g++ $< -o $@ -c

disjointset.o: disjointset.cpp
	g++ $< -o $@ -c

labelTwoPass: labelTwoPass.cpp ppm.o disjointset.o
	g++ $< -o $@ ppm.o disjointset.o