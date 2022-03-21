all: MMult1 val_test01_solved val_test02_solved omp_bug2_solved omp_bug3_solved omp_bug4_solved omp_bug5_solved omp_bug6_solved

MMult1: MMult1.cpp
	g++ -fopenmp -std=c++11 -O3 -march=native MMult1.cpp -o Mult1.cpp

val_test01_solved: val_test01_solved.cpp
	g++ -O3 -std=c++11 -fopenmp val_test01_solved.cpp -o val_test01_solved

val_test02_solved: val_test02_solved.cpp
	g++ -O3 -std=c++11 -fopenmp val_test02_solved.cpp -o val_test02_solved

omp_bug2_solved: omp_bug2_solved.c
	g++ -O3 -std=c++11 -fopenmp $^ -o omp_bug2_solved

omp_bug3_solved: omp_bug3_solved.c
	g++ -O3 -std=c++11 -fopenmp $^ -o omp_bug3_solved

omp_bug4_solved: omp_bug4_solved.c
	g++ -O3 -std=c++11 -fopenmp $^ -o omp_bug4_solved

omp_bug5_solved: omp_bug5_solved.c
	g++ -O3 -std=c++11 -fopenmp $^ -o omp_bug5_solved

omp_bug6_solved: omp_bug6_solved.c
	g++ -O3 -std=c++11 -fopenmp $^ -o omp_bug6_solved

#jacobi2D-omp: jacobi2D-omp.c
#	g++ -O3 -fopenmp $^ -o jacobi2D-omp
#
#gs2D-omp: gs2D-omp.c
#	g++ -O3 -fopenmp $^ -o gs2D-omp


clean:
	rm -f MMult1 val_test01_solved val_test02_solved omp_bug2_solved omp_bug3_solved omp_bug4_solved omp_bug5_solved omp_bug6_solved
