task2: task2.cpp
	mpic++ task2.cpp -o task2

task2_testmode: task2.cpp
	mpic++ -D TEST=1 task2.cpp -o task2

run: task2
	mpirun -np 2 ./task2 20 1 1
	mpirun -np 4 ./task2 20 1 1
	mpirun -np 8 ./task2 20 1 1
	mpirun -np 16 ./task2 20 1 1

test: task2_testmode
	mpirun -np 2 ./task2_testmode 20 1 0 source_vector
	mpirun -np 4 ./task2_testmode 20 1 0 source_vector
	mpirun -np 8 ./task2_testmode 20 1 0 source_vector
	mpirun -np 16 ./task2_testmode 20 1 0 source_vector



