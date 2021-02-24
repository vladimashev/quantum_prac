#include <iostream>
#include <complex>
#include "stdlib.h"
#include "omp.h"
// command line: ./Task1 n k  
// argv             0    1 2
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
    if (argc != 3)
    {
        std::cout << "Wrong number of parameters in command line!";
        return 1;
    }
    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    long long i;
    complexd *U = new complexd[4]; //U[i][j] <-> *(U + i * 2 + j)
    *(U + 1 * 2 + 0) = 0;
    *(U + 0 * 2 + 1) = 1;
    *(U + 1 * 2 + 0) = 1;
    *(U + 1 * 2 + 1) = 0;
    
    long long length = 1 << n;

    complexd *v = new complexd[length];
    
    double norm = 0;
    #pragma omp parallel default(none) private(i) shared(n, k, U, v, length) reduction (+: norm)
    {
        #pragma omp for
        for (i = 0; i < length; ++i)
        {
            srand(omp_get_thread_num() + i);
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
    start = omp_get_wtime();
    OneQubitEvolution(v, new_v, U, length, n, k);
    finish = omp_get_wtime();
    
    finish -= start;
    std::cout << finish;
    
    delete [] U;
    delete [] v;
    delete [] new_v;
    return 0;
}
