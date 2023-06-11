#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
void matrix_vector_multiply(double* A, double* x, double* y, int m, int n,
int k, MPI_Comm comm) {
int rank, size;
MPI_Comm_rank(comm, &rank);
MPI_Comm_size(comm, &size);
int local_n = n / size;
double* local_A = (double*)malloc(m * local_n * sizeof(double));
double* local_x = (double*)malloc(local_n * sizeof(double));
MPI_Scatter(A, m * local_n, MPI_DOUBLE, local_A, m * local_n, MPI_DOUBLE, 
0, comm);
MPI_Scatter(x, local_n, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, 
comm);
double* local_y = (double*)malloc(m * sizeof(double));
for (int i = 0; i < m; i++) {
local_y[i] = 0.0;
for (int j = 0; j < local_n; j++) {
local_y[i] += local_A[i * local_n + j] * local_x[j];
}
}
MPI_Reduce(local_y, y, m, MPI_DOUBLE, MPI_SUM, 0, comm);
free(local_A);
free(local_x);
free(local_y);
}
int main(int argc, char** argv) {
MPI_Init(&argc, &argv);
int rank, size;
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
int m = 3;
int n = 7;
int k = n / size;
double* A = (double*)malloc(m * n * sizeof(double));
double* x = (double*)malloc(n * sizeof(double));
double* y = (double*)malloc(m * sizeof(double));
if (rank == 0) {
for (int i = 0; i < m; i++) {
for (int j = 0; j < n; j++) {
A[i * n + j] = i + j * j +1;
}
}
for (int i = 0; i < n; i++) {
x[i] = i+2*i;
}
}
matrix_vector_multiply(A, x, y, m, n, k, MPI_COMM_WORLD);
if (rank == 0) {
FILE *file = fopen("result.txt", "a");
if (file == NULL) {
return 1;
}
fprintf(file, "Matrix:");
for (int i = 0; i < m * n; i++) {
if(i%n==0){
fprintf(file, "\n");
}
fprintf(file, "%f ", A[i]);
}
fprintf(file, "\n");
fprintf(file, "Vector: \n");
for (int i = 0; i < n; i++) {
fprintf(file, "%f ", x[i]);
}
fprintf(file, "\n");
fprintf(file, "Result: \n");
for (int i = 0; i < m; i++) {
fprintf(file, "%f ", y[i]);
}
fprintf(file, "\n");
fclose(file);
}
free(A);
free(x);
free(y);
MPI_Finalize();
return 0;
}
