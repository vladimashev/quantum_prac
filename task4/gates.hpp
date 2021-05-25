#include <mpi.h>
#include <iostream>
#include <complex>
using namespace std;
typedef complex<double> complexd;

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


void TwoQubitEvolution(complexd *buf0, complexd *buf1, complexd *buf2, complexd *buf3, complexd U[4][4], int n, int k, int l, int rank, int size) 
{
    unsigned N = 1u << n;
    complexd *buf_ans;
    buf_ans = new complexd[size];
    unsigned first_index = rank * size;
    if(l > k)
    {
       	int buffer = k;
       	k = l;
       	l = buffer;
    }
    unsigned rank1_change = first_index ^ (1u << (k - 1)); 
    unsigned rank2_change = first_index ^ (1u << (l - 1));     
	unsigned rank3_change = first_index ^ ( (1u << (k - 1)) | (1u << (l - 1)) ); 

    rank1_change /= size;
    rank2_change /= size;
    rank3_change /= size;
    if (rank == rank1_change) 
    { 
        int shift1 = k - 1;
		int shift2 = l -1;
		int q1 = 1 << shift1;
		int q2 = 1 << shift2;

        for (int i = rank * size; i < rank * size + size; i++) 
        {
        	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;	
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;
			int iq = (iq1 << 1) + iq2;
		
			*(buf_ans + i - rank * size) = 
                *(U + iq * 4 + (0 << 1) + 0] * *(buf0 + first - rank * size) 
                + *(U + iq * 4 + (0 << 1) + 1) * *(buf0 + second - rank * size) 
                + *(U + iq * 4 + (1 << 1) + 0) * *(buf0 + third - rank * size) 
                + *(U + iq * 4 + (1 << 1) + 1) * *(buf0 + fourth - rank * size);
        }
    } else if (rank == rank2_change) 
    { 
        MPI_Sendrecv(buf0, size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, size, MPI_DOUBLE_COMPLEX, rank1_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int shift1 = k - 1;
		int shift2 = l -1;
		int q1 = 1 << shift1;
		int q2 = 1 << shift2;
        for (int i = size * rank; i < rank * size + size; i++) 
        {
        	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;

			int iq=(iq1<<1)+iq2;
			if (first < size * rank || first >= size * (rank + 1))
            {
				first -= rank1_change * size;
				second -= rank1_change * size;
				third -= rank * size;
				fourth -= rank * size;
				*(buf_ans + i - size * rank) = 
                    *(U + iq * 4 + (0 << 1) + 0) * *(buf1 + first) 
                    + *(U + iq * 4 (0 << 1) + 1) * *(buf1 + second) 
                    + *(U + iq * 4 + (1 << 1) + 0) * buf0[third] 
                    + U[iq][(1<<1)+1] * buf0[fourth];
			} else
            {
				third -= rank1_change * size;
				fourth -= rank1_change * size;
				first -= rank * size;
				second -= rank * size;
				*(buf_ans + i - size * rank) = 
                    *(U + iq * 4 + (0 << 1) + 0) * *(buf0 + first) 
                    + *(U + iq * 4 + (0<<1) + 1) * *(buf0 + second) 
                    + *(U + iq * 4 + (1<<1) + 0) * *(buf1 + third) 
                    + *(U + iq * 4 + (1 << 1) + 1) * *(buf1 + fourth);

			}
        }

    } else 
    { 
        MPI_Sendrecv(buf0, size, MPI_DOUBLE_COMPLEX, rank1_change, 0, buf1, size, MPI_DOUBLE_COMPLEX, rank1_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf0, size, MPI_DOUBLE_COMPLEX, rank2_change, 0, buf2, size, MPI_DOUBLE_COMPLEX, rank2_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(buf0, size, MPI_DOUBLE_COMPLEX, rank3_change, 0, buf3, size, MPI_DOUBLE_COMPLEX, rank3_change, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int shift1 = k - 1;
		int shift2 = l - 1;
		int q1 = 1 << shift1;
		int q2 = 1 << shift2;

        for (int i = size * rank; i < size * rank + size; i++) 
        {
           	int first = i & ~q1 & ~q2;
			int second = i & ~q1 | q2;
			int third = (i | q1) & ~q2;
			int fourth = i | q1 | q2;
			int iq1 = (i & q1) >> shift1;
			int iq2 = (i & q2) >> shift2;
			int iq = (iq1 << 1) + iq2;
			*(buf_ans + i - size * rank) = 0;
			if (rank * size <= first && first < rank * size + size)
            {
				first -= rank * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 0) * *(buf0 + first);
			} else if (rank1_change * size <= first && first < rank1_change * size + size)
            {
				first -= rank1_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 0) * *(buf1 + first);
            } else if (rank2_change * size <= first && first < rank2_change * size + size)
            {
            	first -= rank2_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 0) * *(buf2 + first);
			} else if (rank3_change * size <= first && first< rank3_change * size + size)
            {
				first -= rank3_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 0) * *(buf3 + first);
			}


			if (rank * size <= second && second < rank * size + size)
            {
				second -= rank*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 1] * *(buf0 + second);
			} else if (rank1_change *size <= second && second < rank1_change * size + size)
            {
				second -= rank1_change*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 1) * *(buf1 + second);
            } else if (rank2_change *size <= second && second < rank2_change * size + size)
            {
            	second -= rank2_change*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 1) * *(buf2 + second);
			} else if (rank3_change *size <= second && second < rank3_change * size + size)
            {
				second -= rank3_change*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (0 << 1) + 1) * *(buf3 + second);
			}


			if (rank * size <= third && third < rank * size + size)
            {
				third -= rank * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 0) * *(buf0 + third);
			} else if (rank1_change * size <= third && third < rank1_change * size + size)
            {
				third -= rank1_change*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 0) * *(buf1 + third);
            } else if (rank2_change * size <= third && third < rank2_change * size + size)
            {
            	third -= rank2_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 0) * *(buf2 + third);
			} else if (rank3_change * size <= third && third < rank3_change * size + size)
            {
				third -= rank3_change * size;
				*(buf_ans + i - size * rank)+= *(U + iq * 4 + (1 << 1) + 0) * *(buf3 + third);
			}


			if(rank * size <= fourth && fourth < rank * size + size)
            {
				fourth -= rank*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 1) * *(buf0 + fourth);
			} else if (rank1_change * size <= fourth && fourth < rank1_change * size + size)
            {
				fourth -= rank1_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 1) * *(buf1 + fourth);
            } else if (rank2_change * size <= fourth && fourth < rank2_change * size + size)
            {
            	fourth -= rank2_change*size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 1) * *(buf2 + fourth);
			} else if (rank3_change * size <= fourth && fourth < rank3_change * size + size)
            {
				fourth -= rank3_change * size;
				*(buf_ans + i - size * rank) += *(U + iq * 4 + (1 << 1) + 1) * *(buf3 + fourth);
			}
        }
    }
    
    for (int i = 0; i < size; ++i)
    {
    	*(buf0 + i) = *(buf_ans + i);
	}
}


void NOT(int k, complexd *buf0, int rank, int size, int n)
{
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd *U = new complexd[4];
    *(U + 0 * 2 + 0) = 0.0
    *(U + 0 * 2 + 1) = 1.0;
    *(U + 1 * 2 + 0) = 1.0;
    *(U + 1 * 2 + 1) = 0.0;
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);
    delete [] U;
}


void CNOT(int k, int l, complexd *buf0, int rank, int size, int n) 
{
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
     complexd *U = new complexd[16];
    for (int i = 0; i < 4; i++) 
    {
        for (int j = 0; j < 4; j++) 
        {
            *(U + i * 4 + j) = 0.0;
        }
    }
    *(U + 0 * 4 + 0) = 1.0
    *(U + 1 * 4 + 1) = 1.0;
    *(U + 2 * 4 + 3) = 1.0;
    *(U + 3 * 4 + 2) = 1.0;
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
    delete [] U;
}


void H(int k, complexd *buf0, int rank, int size, int n) 
{
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd *U = new complexd[4];
    *(U + 0 * 2 + 0) = 1.0 / sqrt(2);
    *(U + 0 * 2 + 1) = 1.0 / sqrt(2) ;
    *(U + 1 * 2 + 0) = 1.0 / sqrt(2);
    *(U + 1 * 2 + 1) = - 1.0 / sqrt(2);
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);
    delete [] U;
}

void H_n(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) 
{
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
    complexd *U = new complexd[16];
    for (int i = 0; i < 4; ++i) 
    {
        for (int j = 0; j < 4; ++j) 
        {
            *(U + i * 4 + j) = 0.5;
        }
    }
    *(U + 1 * 4 + 1) = -0.5
    *(U + 1 * 4 + 3) = -0.5;
    *(U + 2 * 4 + 2) = -0.5;
    *(U + 2 * 4 + 3) = -0.5;
    *(U + 3 * 4 + 1) = -0.5;
    *(U + 3 * 4 + 2) = -0.5;
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
    delete [] U;
}

void ROT(unsigned k, complexd *buf0, int rank, int size, unsigned n) 
{
    complexd *buf1;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    complexd *U = new complexd[4];
    *(U + 0 * 2 + 0) = 1.0;
    *(U + 0 * 2 + 1) = 0.0;
    *(U + 1 * 2 + 0) = 0.0;
    *(U + 1 * 2 + 1) = - 1.0;
    OneQubitEvolution(buf0, buf1, U, n, k, rank, seg_size);
    delete [] U;
}

void CROT(unsigned k, unsigned l, complexd *buf0, int rank, int size, unsigned n) 
{
    complexd *buf1, *buf2, *buf3;
    unsigned N = 1u << n;
    unsigned seg_size = N / size;
    buf1 = new complexd[seg_size];
    buf2 = new complexd[seg_size];
    buf3 = new complexd[seg_size];
    complexd *U = new complexd[16];
    for (int i = 0; i < 4; ++i) 
    {
        for (int j = 0; j < 4; ++j) 
        {
           *(U + i * 4 + j) = 0.0;
        }
    }
    *(U + 0 * 4 + 0) = 1.0
    *(U + 1 * 4 + 1) = 1.0;
    *(U + 2 * 4 + 2) = 1.0;
    *(U + 3 * 4 + 3) = -1.0;
    TwoQubitEvolution(buf0, buf1, buf2, buf3, U, n, k, l, rank, seg_size);
    delete [] U;
}
