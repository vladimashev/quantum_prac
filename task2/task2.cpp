#include <iostream>
#include <complex>
#include "stdlib.h"
#include <mpi.h>
// command line:  ./Task1 n k mode filename
// argv              0    1 2  3       4
typedef std::complex<double> complexd;


void OneQubitEvolution(complexd *in, complexd *out, complexd *U, long long length, int n, int k)
{
    int shift = n - k;
    int mask = 1 << shift;
    int i;
    #pragma omp parallel default(none) private(i) shared(mask, length, shift, n, k, in, out, U)
    {
        #pragma omp for
        for (i = 0; i < length; ++i)
        {
            int k_bit_to_zero = i & (~ mask);
            int k_bit_to_one = k_bit_to_zero | mask;
            int k_bit = (i & mask) >> shift;
            *(out + i) = *(U + k_bit * 2 + 0) * *(in + k_bit_to_zero) + *(U + k_bit * 2 + 1) * *(in + k_bit_to_one);
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
        MPI_File_open(MPI_COMM_WORLD, argv[4], MPI_MODE_RDONLY, MPI_INFO_NULL, &file); //argv[4] is a name of the file
        
        
        double *buffer = new double[2];
        int shift = size * rank * sizeof(buffer);
        
        MPI_File_seek(file, shift, MPI_SEEK_SET);
        
        for (int i = 0; i < count_of_elements; ++i) 
        {
            MPI_File_read(file, buffer, 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
            elements[i] = complexd(buffer[0], buffer[1]);
        }
        
        MPI_File_close(&file);
        delete [] buffer;
    }
    else //random vector
    {

        double module = 0.0;
        double norm = 0.0;
        for (int i = 0; i < count_of_elements; ++i) 
        {
            srand(rank * count_of_elements + i);
            arr[i] = complexd(rand(), rand());
            double curr_abs = abs(arr[i]);
            sum += curr_abs * curr_abs;
        }

        MPI_Allreduce(&sum, &norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        norm = sqrt(norm);

        for (int i = 0; i < size; ++i) 
        {
            elements[i] = elements[i] / norm;
        }
    }
    
    
    
    
    #pragma omp parallel default(none) private(i) shared(n, k, U, v, length) reduction (+: norm)
    {
        #pragma omp for
        for (i = 0; i < length; ++i)
        {
            srand(i);
            v[i].real(rand());
            v[i].imag(rand());
            norm += abs(v[i]) * abs(v[i]);
        }
    }
    norm = sqrt(norm);
    #pragma omp parallel default(none) private(i) shared(n, k, U, v, norm, length)
    {
        #pragma omp for
        for (i = 0; i < length; ++i)
        {
            v[i] = v[i] / norm;
        }
    }
    /*
    for (i = 0; i < length; ++i)
    {
        std::cout << std::endl << v[i];
    }
    */
    complexd *new_v = new complexd[length];

    double start, finish;

    
    finish -= start;
    std::cout << finish << std::endl;
    
    delete [] U;
    delete [] new_v;
    MPI_Finalize();
    return 0;
}
