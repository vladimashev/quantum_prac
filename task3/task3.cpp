#include <iostream>
#include <complex>
#include "stdlib.h"
#include <mpi.h>
#include <stdio.h>
#define EPS 0.01
#define RESFILE_NO_NOISE "result_no_noise"
#define RESFILE_NOISE "result_noise"

// command line:  ./Task1 n k mode filename (for reading the vector)
// argv              0    1 2  3       4
typedef std::complex<double> complexd;

double normal_dis_gen()
{
    double S = 0.;
    for (int i = 0; i < 12; ++i) 
    { 
        S += (double) rand() / RAND_MAX; 
    }
    return S - 6.0;
}


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
    setbuf(stdout, NULL);
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
    double *buffer = new double[2];
    if (mode == 0) //read vector from the file
    {
        MPI_File data;
        MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_RDONLY, MPI_INFO_NULL, &data); //argv[4] is a name of the file
        
        
        
        int shift = count_of_elements * rank * 2 * sizeof(double);
        
        MPI_File_seek(data, shift, MPI_SEEK_SET);
        
        for (int i = 0; i < count_of_elements; ++i) 
        {
            MPI_File_read(data, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            elements[i] = complexd(buffer[0], buffer[1]);
        }
        
        MPI_File_close(&data);
       
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
    
    
    complexd *new_elements_no_noise = new complexd[count_of_elements];
    complexd *new_elements_noise = new complexd[count_of_elements];
    double start, finish, time_no_noise, time_noise;
    
    //Evolution without noise
    time_no_noise = 0.0;
    for (int k = 1; k <= n; ++k)
    {
        start = MPI_Wtime();
        OneQubitEvolution(elements, new_elements_no_noise, U, n, k, count_of_elements, rank);
        finish = MPI_Wtime();
        time_no_noise += finish - start;
    }
    
    double time;
    MPI_Reduce(&time_no_noise, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        std::cout << "TIME WITHOUT NOISE: " << time << std::endl;    
    }
    MPI_Barrier(MPI_COMM_WORLD);
    //Evolution with noise
    time_noise = 0.0;
    complexd *U_noise = new complexd[4];
    for (int k = 1; k <= n; ++k)
    {
        double theta = 0;
        if (rank == 0) 
        {
            theta = normal_dis_gen();
        }
        theta = theta * EPS;
        MPI_Bcast(&theta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        *(U_noise + 0 * 2 + 0) = *(U + 0 * 2 + 0) * cos(theta) - *(U + 0 * 2 + 1) * sin(theta);
        *(U_noise + 0 * 2 + 1) = *(U + 0 * 2 + 0) * sin(theta) + *(U + 0 * 2 + 1) * cos(theta);
        *(U_noise + 1 * 2 + 0) = *(U + 1 * 2 + 0) * cos(theta) - *(U + 1 * 2 + 1) * sin(theta);
        *(U_noise + 1 * 2 + 1) = *(U + 1 * 2 + 0) * sin(theta) + *(U + 1 * 2 + 1) * cos(theta);
        start = MPI_Wtime();
        OneQubitEvolution(elements, new_elements_noise, U_noise, n, k, count_of_elements, rank);
        finish = MPI_Wtime();
        time_noise += finish - start;

    }
    
    MPI_Reduce(&time_noise, &time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) 
    {
        std::cout << "TIME WITH NOISE: " << time << std::endl;    
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int shift;
    
    MPI_File data;
    MPI_File_open(MPI_COMM_WORLD, RESFILE_NO_NOISE, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &data);
    
    
    /*
    shift = count_of_elements * rank * 2 * sizeof(double);
        
    MPI_File_seek(data, shift, MPI_SEEK_SET);
    for (int i = 0; i < count_of_elements; ++i) {
        buffer[0] = new_elements_no_noise[i].real();
        buffer[1] = new_elements_no_noise[i].imag();
        MPI_File_write(data, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&data);
    
    MPI_File_open(MPI_COMM_WORLD, RESFILE_NOISE, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &data);
    shift = count_of_elements * rank * 2 * sizeof(double);
        
    MPI_File_seek(data, shift, MPI_SEEK_SET);
    for (int i = 0; i < count_of_elements; ++i) {
        buffer[0] = new_elements_noise[i].real();
        buffer[1] = new_elements_noise[i].imag();
        MPI_File_write(data, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&data);
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double Fidelity;
    complexd sum = 0.0;
    complexd tmp;
    //Compute Fidelity as a scalar product 
    for (int i = 0; i < count_of_elements; ++i)
    {
        sum += new_elements_no_noise[i] * conj(new_elements_noise[i]);
    }
    MPI_Reduce(&sum, &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD);
    Fidelity = abs(tmp) * abs(tmp);
    
    if (rank == 0) 
    {
        std::cout << "LOSS: " << 1.0 - Fidelity << std::endl;    
    }
    
    delete [] buffer;
    delete [] U;
    delete [] U_noise;
    delete [] new_elements_noise;
    delete [] new_elements_no_noise;
    delete [] elements;
    
    MPI_Finalize();
    
    return 0;
}
