test: Task1.cpp
	g++ Task1.cpp -o Task1 -fopenmp -fsanitize=address
	OMP_NUM_THREADS=8 ./Task1 20 8
	OMP_NUM_THREADS=8 ./Task1 24 8
	OMP_NUM_THREADS=8 ./Task1 28 8
	OMP_NUM_THREADS=8 ./Task1 30 8
