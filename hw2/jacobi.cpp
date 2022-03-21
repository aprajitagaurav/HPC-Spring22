
#include <stdio.h>
#include <string>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;
int N;

void jacobi(vector <vector <double> > u, vector <vector <double> > f, double h) {
    int k = 0;   
    vector<vector<double>> uOld (N, vector<double> (N));
    int i=0, j=0;
    cout<<u[i][j]<<endl;
    int it;
    for(it=0 ; it<5000 ; it++) {
        
        cout<<"here2"<<endl;
        cout<<"printing u:\n";
        for(int i=0; i<N ; i++) {
                cout<<"hi";
            for(int j=0 ; j<N ; i++) {
                cout<<u[i][j]<<" ";                
                uOld[i][j] = u[i][j];
            }
            cout<<endl;
        }

        for(int i=1 ; i<N-1 ; i++) {
            for(int j=1 ; j<N-1 ; i++) {
                k++;
                double sum = 0;
                        cout<<"here3"<<endl;

                u[i][j] = 0.25 * ((h*h)*(f[i][j]) + uOld[i-1][j] 
                        + uOld[i][j-1] + uOld[i+1][j] + uOld[i][j+1]);
            }
        }

        
        
    }
    
}

int main(int argc, char** argv) {
    
    N = stoi(argv[1]);
    vector<vector<double>> A (N, vector<double> (N));
    for (int i=0 ; i<N ; i++) {
        for (int j=0 ; j<N ; j++) {
            if(i == j)
                A[i][j] = 2.0;
            else if((i == (j-1)) || (i == (j+1))) 
                A[i][j] = -1.0;
            else
                A[i][j] = 0.0;
            A[i][j] *= (N+1) * (N+1);
        }
    }
    cout<<"here"<<endl;

    double h = 1/(N+1);
    vector<vector<double>> u (N, vector<double> (N));
    vector<vector<double>> f (N, vector<double> (N));
    
    
    for(int i=0; i<N ; i++) {
        for(int j=0; j<N ; j++) {
           f[i][j] = 1.0;
            u[i][j] = 0.0;
        }
    }
cout<<"here1"<<endl;
    jacobi(u, f, h);
    
    return 0;
}

