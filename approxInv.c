#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>

void printMat(double *A, int n, int m)
{
    int i, j;
    printf("\nMatrix:\n");
    printf("[ ");
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < m; j++) {
            printf("%f ", A[j+i*m]);
        }
        if (i == n-1)
            printf("]\n");
        else
            printf("]\n[ ");
    }
}

void transpose(double *A, int n_a, int m_a, double *A_t)
{
    int i, j;
    for (i = 0; i < n_a; i++)
        for (j = 0; j < m_a; j++)
            A_t[i+j*n_a] = A[j+i*m_a];
    return;
}

void matMul(double *A, int n, int m, double *B, int k, double *P)
{
    /*
    int i, j, k;
    // double *B_t = malloc(m_b*n_b*sizeof(double));
    // transpose(B, n_b, m_b, B_t);
    memset(P, 0, n_a*m_b*sizeof(double));
    for (i = 0; i < n_a; i++)
        for (j = 0; j < m_b; j++)
            for (k = 0; k < m_a; k++)
                // P[j+i*m_b] += A[k+i*m_a] * B[j+k*m_b];
                // P[j+i*m_b] += A[k+i*m_a] * B_t[k+j*m_a];
                P[j+i*m_b] += A[k+i*m_a] * B[k+j*m_a];
    // free(B_t);
    return;
    */
   // cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, m, k, 1.0, A, k, B, m, 0.0, P, n);
   cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, n, m, 1.0, A, n, B, m, 0.0, P, n);
   // P = 1.0*A.dot(B) +0.0*P
   return;
}

void matAdd(double *A, int n_a, int m_a, double *B, double *S)
{
    int i, j;
    for (i = 0; i < n_a; i++)
        for (j = 0; j < m_a; j++)
                S[j+i*m_a] = A[j+i*m_a] + B[j+i*m_a];
    return;
}

void matSub(double *A, int n_a, int m_a, double *B, double *S)
{
    int i, j;
    for (i = 0; i < n_a; i++)
        for (j = 0; j < m_a; j++)
                S[j+i*m_a] = A[j+i*m_a] - B[j+i*m_a];
    return;
}

void scalMul(double *A, int n, int m, double s, double *A_s)
{
    /*
    int i, j;
    for (i = 0; i < n_a; i++)
        for (j = 0; j < m_a; j++)
            A_s[j+i*m_a] = s * A[j+i*m_a];
    return;
    */
   memcpy(A_s, A, n*m*sizeof(double));
   cblas_dsymm(CblasRowMajor, CblasLeft, CblasLower, n, m, 0.0, A, n, A, n, s, A_s, n);
   // A_s = A
   // A_s = 0.0*A.dot(B) +s*A_s
   return;
}

void sumColAbs(double *A, int n, int m, double *B)
{
    int i, j;
    memset(B, 0, n*sizeof(double));
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++)
            B[i] += fabs(A[j+i*m]);
    return;
}

double maxVec(double *A, int n)
{
    int i;
    double a_max = -1e-31;
    for (i = 0; i < n; i++)
        if (A[i] > a_max)
            a_max = A[i];
    return a_max;
}

double fNorm1(double *A, int n_a, int m_a)
{
    int i, j;
    double arg, ret = 0;
    for (i = 0; i < n_a; i++)
    {
        for (j = 0; j < m_a; j++) {
            arg = A[j+i*m_a];
            ret += (arg * arg);
        }
    }
    ret = sqrt(ret);
    return ret;
}

double fNorm2(double *A, int n_a, int m_a, double *B)
{
    int i, j;
    double arg, ret = 0;
    for (i = 0; i < n_a; i++)
    {
        for (j = 0; j < m_a; j++) {
            arg = A[j+i*m_a]-B[j+i*m_a];
            ret += (arg * arg);
        }
    }
    ret = sqrt(ret);
    return ret;
}

void setId(int n, double *I)
{
    int i;
    memset(I, 0, n*n*sizeof(double));
    for (i = 0; i < n; i++)
        I[i+i*n] = 1.0;
    return;
}

int approxInv(double *A, int n, int order, double tol, double *A_i)
{
    double alpha, fN;
    // double I[n*n], A_t[n*n], P[n*n], T[n*n], Y[n*n], Z[n*n], C[n];
    double *I = malloc(n*n*sizeof(double));
    double *A_t = malloc(n*n*sizeof(double));
    double *P = malloc(n*n*sizeof(double));
    double *T = malloc(n*n*sizeof(double));
    double *Y = malloc(n*n*sizeof(double));
    double *Z = malloc(n*n*sizeof(double));
    double *C = malloc(n*sizeof(double));
    // int i, j, m = n;
    int i, m = n;
    // Set identity
    setId(n, I);
    // Initialize inverse
    // transpose(A, n, m, A_t);
    memcpy(A_t, A, n*m*sizeof(double));
    matMul(A, n, m, A_t, n, T);
    sumColAbs(T, n, n, C);
    alpha = maxVec(C, n);
    scalMul(A_t, m, n, 2.0/alpha, T);
    memcpy(A_i, T, n*m*sizeof(double));
    // Zero previous matrix
    memset(P, 0, n*m*sizeof(double));

    // j = 0;
    matMul(A, m, n, A_i, n, T);
    matSub(I, m, n, T, Y);
    fN = fNorm1(Y, m, n);
    // while (fNorm2(A_i, m, n, P) > tol)
    while (fN > tol)
    {
        // j += 1;
        // memcpy(P, A_i, n*m*sizeof(double));
        // matMul(A, m, n, A_i, n, m, T);
        // matSub(I, m, n, T, Y);
        matAdd(I, m, n, Y, Z);
        for (i = 0; i < order-1; i++)
        {
            matMul(Y, m, n, Z, n, T);
            matAdd(I, m, n, T, Z);
        }
        matMul(A_i, m, n, Z, n, T);
        memcpy(A_i, T, n*m*sizeof(double));
        matMul(A, m, n, A_i, n, T);
        matSub(I, m, n, T, Y);
        fN = fNorm1(Y, m, n);
    }
    free(I);
    free(A_t);
    free(P);
    free(T);
    free(Y);
    free(Z);
    free(C);
    return 1;
}
