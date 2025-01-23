#include"matrice.h"
#include<stdio.h>
#include"neural_network_tests.cu"
void goofytest(){
    matrice* A=random_mat(5,1,10);
    matrice* B = random_mat(3,3,10);
    matrice* C = transpose(A);
    // sum(A,B,C);

    print_mat(A);
    // mat_SOFT_MAX(A,C);
    // print_mat(B);
    print_mat(C);
}

void neural_test(){
    
}

int main(int argc, char const *argv[])
{

    train_and_test_MNIST();
    return 0;
}
