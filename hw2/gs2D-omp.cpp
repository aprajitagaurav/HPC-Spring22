#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "utils.h"
using namespace std;

void gaussSeidel(int N, double *u, double *f, int NUM_THREADS) {    
    double* uOld = (double *)calloc(sizeof(double), (N+2)*(N+2));
    double h = 1 / (N + 1.0);

    int k = 0;
    double result = 1;
    for(int it=0 ; it<5000 ; it++) {
        if(result > 1e-6) {
            double sum1 = 0; double sum2 = 0;

            #ifdef _OPENMP
            #pragma omp parallel for collapse(2) reduction(+:sum1) num_threads(NUM_THREADS)
            #endif
            for (int i = 1; i < N + 1; i++) {
                for (int j = 1; j < N + 1; j+=2) {
                    double residue;
                    int idx = (N + 2)*i + j;
                    
                    if (i%2 == 1)
                        idx++;
                                        
                    uOld[idx] = h*h*f[idx] + u[idx - 1] + u[idx + 1] + u[idx - (N + 2)] + u[idx + (N + 2)];
                    uOld[idx] = 0.25*uOld[idx];

                    residue = (1 / (h*h))*(4 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - (N + 2)] - u[idx + (N + 2)]) - f[idx];
                    sum1 += residue*residue;
                }
            }

            #ifdef _OPENMP
            #pragma omp flush(uOld)
            #pragma omp parallel for collapse(2) reduction(+:sum2) num_threads(NUM_THREADS)
            #endif
            for (int i = 1; i < N + 1; i++) {
                for (int j = 1; j < N + 1; j+=2) {
                    double residue;

                    int idx = (N + 2)*i + j;
                    if (i%2 == 0)
                        idx++;
                    
                    uOld[idx] = h*h*f[idx] + uOld[idx - 1] + uOld[idx + 1] + uOld[idx - (N + 2)] + uOld[idx + (N + 2)];
                    uOld[idx] = 0.25 * uOld[idx];
                    
                    residue = (1 / (h*h))*(4 * u[idx] - u[idx - 1] - u[idx + 1] - u[idx - (N + 2)] - u[idx + (N + 2)]) - f[idx];
                    sum2 += residue*residue;
                }
            }
            #ifdef _OPENMP
            #pragma omp flush(uOld)
            #endif
            result = sqrt(sum1+sum2);
            cout<<"residue "<<it+1<<" : "<< result <<endl;
            double *uTemp = uOld;
            uOld = u;
            u = uTemp;
        }
    }
}

int main(int argc, char** argv) {
    int max_itr = 100;

    for (int pwr = 0; pwr < 4; pwr++) {

        int N = 2 * pow(10, pwr) + 5;
        int NUM_THREADS = 2 * (pwr+1);
        cout<<"N: "<<N<<", NUM_THREADS: "<<NUM_THREADS<<endl;

        double* u = (double *)calloc(sizeof(double), (N + 2)*(N + 2));
        double* f = (double*)malloc((N + 2)*(N + 2) * sizeof(double));
        
        for (int j = 0; j < (N + 2)*(N + 2); j++) {
            f[j] = 1.0;
            u[j] = 0.0;
        }
        
        Timer t;
        t.tic();
        gaussSeidel(N, u, f, NUM_THREADS);
        double time = t.toc();
        cout<<"time: "<<time<<endl<<endl;

        free(f);
        free(u);
    }

    return 0;
}
