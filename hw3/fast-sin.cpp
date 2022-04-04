#include <math.h>
#include <stdio.h>
#include "utils.h"
#include "intrin-wrapper.h"

// Headers for intrinsics
#ifdef __SSE__
#include <xmmintrin.h>
#endif
#ifdef __SSE2__
#include <emmintrin.h>
#endif
#ifdef __AVX__
#include <immintrin.h>
#endif

using namespace std;

// coefficients in the Taylor series expansion of sin(x)
static constexpr double c3  = -1/(((double)2)*3);
static constexpr double c5  =  1/(((double)2)*3*4*5);
static constexpr double c7  = -1/(((double)2)*3*4*5*6*7);
static constexpr double c9  =  1/(((double)2)*3*4*5*6*7*8*9);
static constexpr double c11 = -1/(((double)2)*3*4*5*6*7*8*9*10*11);
// sin(x) = x + c3*x^3 + c5*x^5 + c7*x^7 + x9*x^9 + c11*x^11

void sin4_reference(double* sinx, const double* x) {
  for (long i = 0; i < 4; i++) 
    sinx[i] = sin(x[i]);
}

void sin4_taylor(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x3  = x1 * x2;
    double x5  = x3 * x2;
    double x7  = x5 * x2;
    double x9  = x7 * x2;
    double x11 = x9 * x2;

    double s = x1;
    s += x3  * c3;
    s += x5  * c5;
    s += x7  * c7;
    s += x9  * c9;
    s += x11 * c11;
    sinx[i] = s;
  }
}

void sin4_taylor1(double* sinx, const double* x) {
  for (int i = 0; i < 4; i++) {
    double x1  = x[i];
    double x2  = x1 * x1;
    double x4  = x2 * x2;
    double x6  = x4 * x2;
    double x8  = x6 *  x2;
    double x10 = x8 * x2;
    double x12 = x10 * x2;
    double x14 = x12 * x2;
    //double x16 = x12 * x2;
    //double x18 = x12 * x2;
    //double x20 = x12 * x2;

    static constexpr double c2  = -1/(((double)2));
    static constexpr double c4  =  1/(((double)2)*3*4);
    static constexpr double c6  = -1/(((double)2)*3*4*5*6);
    static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
    static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
    static constexpr double c12 = 1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
    static constexpr double c14 = -1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13*14);
    //static constexpr double c16 = -1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16);
    //static constexpr double c18 = -1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17*18);
    //static constexpr double c20 = -1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13*14*15*16*17*18*19*20);

    double s = 1;
    s += x2  * c2;
    s += x4  * c4;
    s += x6  * c6;
    s += x8  * c8;
    s += x10 * c10;
    s += x12 * c12;
    s += x14 * c14;
    //s += x16 * c16;
    //s += x18 * c18;
    //s += x20 * c20;
    sinx[i] = s;
  }
}

void sin4_intrin(double* sinx, const double* x) {
  // The definition of intrinsic functions can be found at:
  // https://software.intel.com/sites/landingpage/IntrinsicsGuide/#
 // cout<<"In instrinsic fn\n";
#if defined(__AVX__)
  //cout<<"AVX"; 
 __m256d x1, x2, x3, x5, x7, x9, x11;
  x1  = _mm256_load_pd(x);
  x2  = _mm256_mul_pd(x1, x1);
  x3  = _mm256_mul_pd(x1, x2);
  x5  = _mm256_mul_pd(x3, x2);
  x7  = _mm256_mul_pd(x5, x2);
  x9  = _mm256_mul_pd(x7, x2);
  x11 = _mm256_mul_pd(x9, x2);

  __m256d s = x1;
  s = _mm256_add_pd(s, _mm256_mul_pd(x3 , _mm256_set1_pd(c3 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x5 , _mm256_set1_pd(c5 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x7 , _mm256_set1_pd(c7 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x9 , _mm256_set1_pd(c9 )));
  s = _mm256_add_pd(s, _mm256_mul_pd(x11 , _mm256_set1_pd(c11 )));
  _mm256_store_pd(sinx, s);
#elif defined(__SSE2__)
 // cout<<"SSE";
  constexpr int sse_length = 1;
  for (int i = 0; i < 4; i+=sse_length) {
    __m128d x1, x2, x3,  x5, x7, x9, x11;
    x1  = _mm_load_pd(x+i);
    x2  = _mm_mul_pd(x1, x1);
    x3  = _mm_mul_pd(x1, x2);
    x5  = _mm_mul_pd(x3, x2);
    x7  = _mm_mul_pd(x5, x2);
    x9  = _mm_mul_pd(x7, x2);
    x11 = _mm_mul_pd(x9, x2);

    __m128d s = x1;
    s = _mm_add_pd(s, _mm_mul_pd(x3 , _mm_set1_pd(c3 )));
    s = _mm_add_pd(s, _mm_mul_pd(x5 , _mm_set1_pd(c5 )));
    s = _mm_add_pd(s, _mm_mul_pd(x7 , _mm_set1_pd(c7 )));
    s = _mm_add_pd(s, _mm_mul_pd(x9 , _mm_set1_pd(c9 )));
    s = _mm_add_pd(s, _mm_mul_pd(x11 , _mm_set1_pd(c11 )));
    _mm_store_pd(sinx+i, s);
  }
#else
  sin4_reference(sinx, x);
#endif
}

void sin4_vector(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x3;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x3  = x1 * x2;

  Vec4 s = x1;
  s += x3  * c3 ;
  s.StoreAligned(sinx);
}

void sin4_vector1(double* sinx, const double* x) {
  // The Vec class is defined in the file intrin-wrapper.h
  typedef Vec<double,4> Vec4;
  Vec4 x1, x2, x4, x6,x8, x10, x12, x14;
  x1  = Vec4::LoadAligned(x);
  x2  = x1 * x1;
  x4  = x2 * x2;
  x6  = x4 * x2;
  x8  = x6 *  x2;
  x10 = x8 * x2;
  x12 = x10 * x2;
  x14 = x12 * x2;

  static constexpr double c2  = -1/(((double)2));
  static constexpr double c4  =  1/(((double)2)*3*4);
  static constexpr double c6  = -1/(((double)2)*3*4*5*6);
  static constexpr double c8  =  1/(((double)2)*3*4*5*6*7*8);
  static constexpr double c10 = -1/(((double)2)*3*4*5*6*7*8*9*10);
  static constexpr double c12 = 1/(((double)2)*3*4*5*6*7*8*9*10*11*12);
  static constexpr double c14 = -1/(((double)2)*3*4*5*6*7*8*9*10*11*12*13*14);

  Vec4 s = 1;
  s += x2  * c2;
  s += x4  * c4;
  s += x6  * c6;
  s += x8  * c8;
  s += x10 * c10;
  s += x12 * c12;
  s += x14 * c14;
  s.StoreAligned(sinx);
}

double err(double* x, double* y, long N) {
  double error = 0;
  for (long i = 0; i < N; i++) error = std::max(error, fabs(x[i]-y[i]));
  return error;
}

int main() {
  Timer tt;
  long N = 10;
  double* x = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_intrin = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector = (double*) aligned_malloc(N*sizeof(double));

  double* x1 = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_ref1 = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_taylor1 = (double*) aligned_malloc(N*sizeof(double));
  double* sinx_vector1 = (double*) aligned_malloc(N*sizeof(double));

  
  for (long i = 0; i < N; i++) {
    x[i] = (drand48()-0.5) * M_PI/2; // [-pi/4,pi/4]
    //outside 
    x1[i] = x[i] + M_PI/2; 
    //cout<<"x[i]="<<x[i]<<"   x1[i]="<<x1[i]<<endl;
    sinx_ref[i] = 0;
    sinx_taylor[i] = 0;
    sinx_intrin[i] = 0;
    sinx_vector[i] = 0;
  }

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref+i, x+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor(sinx_taylor+i, x+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_taylor, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_intrin(sinx_intrin+i, x+i);
    }
  }
  printf("Intrin time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_intrin, N));

  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector(sinx_vector+i, x+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref, sinx_vector, N));

  //extra credit
  printf("\n-----------Calculations for outside [-pi/2, pi/2]-----------\n");
  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_reference(sinx_ref1+i, x1+i);
    }
  }
  printf("Reference time: %6.4f\n", tt.toc());


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_taylor1(sinx_taylor1+i, x1+i);
    }
  }
  printf("Taylor time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref1, sinx_taylor1, N));


  tt.tic();
  for (long rep = 0; rep < 1000; rep++) {
    for (long i = 0; i < N; i+=4) {
      sin4_vector1(sinx_vector1+i, x1+i);
    }
  }
  printf("Vector time:    %6.4f      Error: %e\n", tt.toc(), err(sinx_ref1, sinx_vector1, N));


  aligned_free(x);
  aligned_free(sinx_ref);
  aligned_free(sinx_taylor);
  aligned_free(sinx_intrin);
  aligned_free(sinx_vector);
}

