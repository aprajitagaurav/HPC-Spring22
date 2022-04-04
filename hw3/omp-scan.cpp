#include <algorithm>
#include <stdio.h>
#include<iostream>
#include <math.h>
#include <omp.h>
using namespace std;
int NUM_THREADS = 8; 

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
//[2, 5, 7]
//[0, 2, 5]
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {  
  cout<<"N: "<<n<<endl<<"Number of threads: "<<NUM_THREADS<<endl;
  long *sumA;
  #pragma omp parallel num_threads(NUM_THREADS)
  {
      int ithread = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      sumA = (long*)malloc(sizeof(long) * (nthreads+1));
      //cout<<"omp_get_num_threads: "<<nthreads<<endl;
      
      #pragma omp single
      prefix_sum[0] = 0;

      long sum = 0;
      #pragma omp for schedule(static) nowait
      for (int i=1; i<n; i++) {
        sum += A[i-1];
        prefix_sum[i] = sum;
      }
      sumA[ithread+1] = sum;

      #pragma omp barrier

      long offset = 0;
      for(int i=0; i<(ithread+1); i++) 
        offset += sumA[i];

      #pragma omp for schedule(static)
      for (int i=1; i<n; i++)
        prefix_sum[i] += offset;
  }
}

int main() {
  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  long start=1;
  for (long i = 0; i < N; i++) 
  {
    A[i] = rand();
  }

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);
 
  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
