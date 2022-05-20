#ifndef MKL_FC_h__
#define MKL_FC_h__

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <math.h>
#include <mkl/mkl.h>
using namespace std;



void mkl_fullyconnected(float *A,float *B,float *C,float *D,int m,int k,int n){
    float alpha=1;
    float beta=1;
    float *K;
    float *L;
    float *M;

    K = (float *)mkl_malloc( m*k*sizeof( float ), 64 );
   
    for(int i=0;i<m*k;i++){
        K[i]=A[i];
    }
    
    L = (float *)mkl_malloc( k*n*sizeof( float ), 64 );
    M = (float *)mkl_malloc( m*n*sizeof( float ), 64 );

    for(int i=0;i<n*k;i++){
        L[i]=B[i];
    }
    
    for(int i=0;i<n*m;i++){
        M[i]=C[i];
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,m, n, k, alpha, K, k, L, n, beta, M, n);
    
    for(int i=0; i<m*n; i++)D[i]=M[i];//Copying values to output matrix.
}
#endif