#include <iostream>
#include <complex>
#include "stdlib.h"
#include <mpi.h>

#define EPS 10e-5
#define RESFILE "result"
// command line:  ./Task1 n k mode filename (for reading the vector)
// argv              0    1 2  3       4
typedef std::complex<double> complexd;


void OneQubitEvolution(complexd *in, complexd *out, complexd *U, int n, int k, long long size, int rank)
{
    //size here is a count of elements per process
    int first_bit = rank * size;
    int shift = n - k;
    int mask = 1 << shift; 
    int rank_inverse_bit = (first_bit ^ mask) / size; //rank of the process with inverse k-th bit

    if (rank == rank_inverse_bit) 
    {
        for (int i = 0; i < size; ++i) 
        {
            int k_bit_to_zero = i & (~ mask);
            int k_bit_to_one = k_bit_to_zero | mask;
            int k_bit = (i & mask) >> shift;
            *(out + i) = *(U + k_bit * 2 + 0) * *(in + k_bit_to_zero) + *(U + k_bit * 2 + 1) * *(in + k_bit_to_one);
        }
        
    } 
    else 
    {
        MPI_Sendrecv(in, size, MPI_DOUBLE_COMPLEX, rank_inverse_bit, 0, out, size, MPI_DOUBLE_COMPLEX, rank_inverse_bit, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int k_bit = (first_bit & mask) >> shift;
        if (k_bit == 1) 
        { 
            for (int i = 0; i < size; ++i) 
            {
                *(out + i) = *(U + k_bit * 2 + 0) * *(out + i) + *(U + k_bit * 2 + 1) * *(in + i);
            }
        } 
        else 
        {
            for (int i = 0; i < size; ++i) 
            {
                *(out + i) = *(U + k_bit * 2 + 0) * *(in + i) + *(U + k_bit * 2 + 1) * *(out + i);
            }
        }
    }
    return;
}

int main(int argc, char *argv[])
{
    if  (((argc == 4) || (atoi(argv[3]) == 1)) && ((argc == 5) || (atoi(argv[3]) == 0)))
    {
        std::cout << "Wrong number of parameters in command line!";
        return 1;
    }
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int mode = atoi(argv[3]);
    long long i;
    complexd *U = new complexd[4]; //U[i][j] <-> *(U + i * 2 + j)
    *(U + 0 * 2 + 0) = 1.0 / sqrt(2);
    *(U + 0 * 2 + 1) = 1.0 / sqrt(2) ;
    *(U + 1 * 2 + 0) = 1.0 / sqrt(2);
    *(U + 1 * 2 + 1) = - 1.0 / sqrt(2);
    long long length = 1 << n;
    double norm = 0;
    
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    long long count_of_elements = length / size;
    complexd *elements = new complexd[count_of_elements];
    if (mode == 0) //read vector from the file
    {
        MPI_File data;
        MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_RDONLY, MPI_INFO_NULL, &data); //argv[4] is a name of the file
        
        
        double *buffer = new double[2];
        int shift = count_of_elements * rank * sizeof(buffer);
        
        MPI_File_seek(data, shift, MPI_SEEK_SET);
        
        for (int i = 0; i < count_of_elements; ++i) 
        {
            MPI_File_read(data, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            elements[i] = complexd(buffer[0], buffer[1]);
        }
        
        MPI_File_close(&data);
        delete [] buffer;
    }
    else //random vector
    {

        double module = 0.0;
        double norm = 0.0;
        double sum = 0.0;
        for (int i = 0; i < count_of_elements; ++i) 
        {
            srand(rank * count_of_elements + i);
            elements[i] = complexd(rand(), rand());
            double curr_abs = abs(elements[i]);
            sum += curr_abs * curr_abs;
        }

        MPI_Allreduce(&sum, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        norm = sqrt(norm);

        for (int i = 0; i < count_of_elements; ++i) 
        {
            elements[i] = elements[i] / norm;
        }
    }
    
    
    complexd *new_elements = new complexd[count_of_elements];

    double start, finish;
    start = MPI_Wtime();
    OneQubitEvolution(elements, new_elements, U, n, k, count_of_elements, rank);
    finish = MPI_Wtime();
    finish -= start;
    int shift;
    
    #ifdef TEST
    
        int correct_flag = 0;
        //read right vector from the file
        complexd *right_elements = new complexd[count_of_elements];
        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, "testdata", MPI_MODE_RDONLY, MPI_INFO_NULL, &file); 
        
        double *testbuffer = new double[2];
        shift = count_of_elements * rank * sizeof(testbuffer);
        
        MPI_File_seek(file, shift, MPI_SEEK_SET);
        
        for (int i = 0; i < count_of_elements; ++i) 
        {
            MPI_File_read(file, testbuffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            right_elements[i] = complexd(testbuffer[0], testbuffer[1]);
        }
        
        MPI_File_close(&file);
        delete [] testbuffer;
        
        //compare two vectors
        int is_right = 1;
        
        for (int i = 0; i < count_of_elements; ++i)
        {
            if (fabs(new_elements[i].real() - right_elements[i].real()) > EPS || fabs(new_elements[i].imag() - right_elements[i].imag()) > EPS)
            {
                is_right = 0;
                break;
            }
        }
        MPI_Reduce(&is_right, &correct_flag, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if ((rank == 0) && (correct_flag == size))
        {
            std::cout << "CORRECT RESULT!" << std::endl;
        }
        else if (rank == 0)
        {
            std::cout << "WRONG RESULT!" << correct_flag << std::endl;
        }
        
    #endif
    
    MPI_File data;
    MPI_File_open(MPI_COMM_WORLD, RESFILE, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &data);
    
    double *buffer = new double[2];
    shift = count_of_elements * rank * sizeof(buffer);
        
    MPI_File_seek(data, shift, MPI_SEEK_SET);
    for (int i = 0; i < count_of_elements; ++i) {
        buffer[0] = new_elements[i].real();
        buffer[1] = new_elements[i].imag();
        MPI_File_write(data, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&data);
    delete [] buffer;
    
    double time;
    MPI_Reduce(&finish, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        std::cout << "TIME: " << time << std::endl;    
    }

    
    delete [] U;
    delete [] new_elements;
    delete [] elements;

    MPI_Finalize();
    return 0;
}
