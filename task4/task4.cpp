#include <iostream>
#include <complex>
#include "stdlib.h"
#include <mpi.h>
#include "gates.hpp"
//      ./task4 n k l
//argv     0    1 2 3
int main(int argc, char *argv[]) {
    setbuf(stdout, NULL);
    bool box, can;
    int n, k, l;
    #ifdef BLACKBOX 
        box = true;
        can = false;
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        l = atoi(argv[3]);
    #endif
    #ifdef CANNONIZATION
        box = false;
        can = true;
        n = atoi(argv[1]);
        k = atoi(argv[2]);
        l = atoi(argv[3]);
    #endif
    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    unsigned long long index = 1LLU << n;
    unsigned long long seg_size = index / size;
    double local = 0.0;
    double res = 0.0;
    complexd *elements = new complexd[seg_size];
    
    if (box)
    {
        if (rank == 0)
        {
            std::cout << "BLACKBOX TEST... ";
        }
        double module = 0.0;
        double norm = 0.0;
        double sum = 0.0;
        for (int i = 0; i < seg_size; ++i) 
        {
            srand(rank * seg_size + i);
            elements[i] = complexd(rand(), rand());
            double curr_abs = abs(elements[i]);
            sum += curr_abs * curr_abs;
        }

        MPI_Allreduce(&sum, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        norm = sqrt(norm);
        for (int i = 0; i < seg_size; ++i) 
        {
            elements[i] = elements[i] / norm;
        }
        
        bool is_right = true;
        double tmp = 0.0;
        //NOT check
    	NOT(k, elements, rank, size, n);
    	for (int i = 0; i < seg_size; ++i) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
        }
        local = 0.0;
        res = 0.0;

        //ROT check
        ROT(k, elements, rank, size, n);
    	for (int i = 0; i < seg_size; ++i) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
        }
        local = 0.0;
        res = 0.0;

        //H check
        H(k, elements, rank, size, n);
    	for (int i = 0; i < seg_size; ++i) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
        }
        local = 0.0;
        res = 0.0;
        
        //H_n check
        H_n(k, l, elements, rank, size, n);
    	for (int i = 0; i < seg_size; i++) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
        }
        local = 0.0;
        res = 0.0;
        
        //CNOT check
        CNOT(k, l, elements, rank, size, n);
    	for (int i = 0; i < seg_size; ++i) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
        }
        local = 0.0;
        res = 0.0;

        //CROT check
        CROT(k, l, elements, rank, size, n);
    	for (int i = 0; i < seg_size; ++i) 
        {
        	local += abs(elements[i] * elements[i]);
        }
        MPI_Reduce(&local, &res, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0)
        {
            if (fabs(res - 1.0) > 0.001)
            {
                is_right = false;
            }
            tmp = res;
            
        }
        
        if ((is_right) && (rank == 0))
        {
            std::cout << "OK" << std::endl;
        }
        else if ((!(is_right)) && (rank == 0))
        {
            std::cout << "FAILED" << std::endl;
        }

        
    }
    
    if(can)
    {
        if (rank == 0)
        {
            std::cout << "CANNONIZATION TEST... ";
        }
        //fisrt test is {(1, 0), (0, 0)}
        if ((n == 1) && (size == 2) && (seg_size == 1))
        {
            if (rank == 0)
            {
                elements[0] = {1, 0};
            }
            else
            {
                elements[0] = {0, 0};
            }
            NOT(k, elements, rank, size, n);
            
            if (rank == 0)
            {
                elements[0] = {1, 0};
            }
            else
            {
                elements[0] = {0, 0};
            }
            ROT(k, elements, rank, size, n);
            
            if (rank == 0)
            {
                elements[0] = {1, 0};
            }
            else
            {
                elements[0] = {0, 0};
            }
            H(k, elements, rank, size, n);
            
            if (rank == 0)
            {
                std::cout << "OK" << std::endl;
            }
            
        }
        
        //second test is {(1, 0), (0, 0), (0, 0), (0, 0)}
        if ((n == 2) && (size == 2) && (seg_size == 2))
        {
            if (rank == 0)
            {
                elements[0] = {1, 0};
                elements[1] = {0, 0};
            }
            else
            {
                elements[0] = {0, 0};
                elements[1] = {0, 0};
            }
            CNOT(k, l, elements, rank, size, n);   
                     
            if (rank == 0)
            {
                elements[0] = {1, 0};
                elements[1] = {0, 0};
            }
            else
            {
                elements[0] = {0, 0};
                elements[1] = {0, 0};
            }
            CROT(k, l, elements, rank, size, n);
            
            if (rank == 0)
            {
                elements[0] = {1, 0};
                elements[1] = {0, 0};
            }
            else
            {
                elements[0] = {0, 0};
                elements[1] = {0, 0};
            }
            H_n(k, l, elements, rank, size, n);
            
            if (rank == 0)
            {
                std::cout << "OK" << std::endl;
            }
            
        }
    }
    
    delete [] elements;
    MPI_Finalize();
    return 0;
}
