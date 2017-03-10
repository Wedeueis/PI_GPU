all: main

clean: 
	rm -f main
	rm -f *.o

labeling.o: labeling.cpp
	g++ $< -o $@ -c

main: main.cpp labeling.o
	g++ $< -o $@ labeling.o
