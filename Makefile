
# CFLAG:-I ./include -std=c++11 -o ./obj
target=main

main:./obj/main.o ./obj/map.o ./obj/util.o ./obj/cuda.o
	nvcc -o main ./obj/main.o ./obj/map.o ./obj/util.o ./obj/cuda.o -I ./include --std=c++11
./obj/main.o:./src/main.cc
	nvcc -c ./src/main.cc -o ./obj/main.o -I ./include --std=c++11
./obj/map.o:./src/map.cc
	g++ -c ./src/map.cc -o ./obj/map.o -I ./include --std=c++11
./obj/util.o:./src/util.cc
	g++ -c ./src/util.cc -o ./obj/util.o -I ./include --std=c++11
./obj/cuda.o:./src/cuda.cu
	nvcc -c ./src/cuda.cu -o ./obj/cuda.o -I ./include --std=c++11

.PHONY:clean
clean:
	rm -r ./obj/*

# gcc -c ./src/util.ccc -o ./obj/util.o -I ./include -std=c++11