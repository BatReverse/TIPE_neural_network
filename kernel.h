#ifndef KERNEL
#define KERNEL
__global__ void cuda_cout(float* A, float* B, int taille, float* res);


__global__ void cuda_adam_apply_momentum(float* M,float Beta1,float* grad,int N);
__global__ void cuda_adam_apply_speed(float* V,float Beta2,float* grad,int N);
__global__ void cuda_adam_apply_hat(float* Vc,float* V, float Betat,int N);
__global__ void cuda_adam_apply_weight(float* W,float learning_rate,float* Mc,float* Vc,float eplsilon,int N);
#endif