#ifndef KERNEL
#define KERNEL
__global__ void cuda_cout(float* A, float* B, int taille, float* res);
#endif