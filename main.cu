#include"matrice.h"
#include<stdio.h>


int main(int argc, char const *argv[])
{
    srand(time(NULL));
    matrice* A=random_mat(3,3,10);
    matrice* B = random_mat(3,3,10);
    matrice* C = copy(A);
    // sum(A,B,C);

    print_mat(A);
    // print_mat(B);
    print_mat(C);

    return 0;
}
