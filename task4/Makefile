test: gates.hpp task4.cpp
	mpic++ -D BLACKBOX task4.cpp -o task4
	mpiexec -n 2 ./task4 2 1 2
	mpiexec -n 4 ./task4 2 1 2
	mpiexec -n 2 ./task4 4 1 2
	mpiexec -n 4 ./task4 4 1 2
	mpiexec -n 2 ./task4 4 1 3
	mpiexec -n 4 ./task4 4 1 3
	mpiexec -n 2 ./task4 4 2 3
	mpiexec -n 4 ./task4 4 2 3
	mpic++ -D CANNONIZATION task4.cpp -o task4_can
	mpiexec -n 2 ./task4_can 1 1 2
	mpiexec -n 2 ./task4_can 2 1 2
	mpiexec -n 2 ./task4_can 2 2 1
