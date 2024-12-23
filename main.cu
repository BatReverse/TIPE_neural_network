#include"matrice.h"
#include<stdio.h>

void goofytest(){
    matrice* A=random_mat(5,1,10);
    matrice* B = random_mat(3,3,10);
    matrice* C = zeros(5,1);
    // sum(A,B,C);

    print_mat(A);
    mat_SOFT_MAX(A,C);
    // print_mat(B);
    print_mat(C);
}

void neural_test(){
    
}

int main(int argc, char const *argv[])
{
    srand(time(NULL));


    return 0;
}
