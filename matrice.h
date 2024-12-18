#ifndef MATRICE
#define MATRICE
struct matrice
{
    int lignes;
    int colonnes;
    double* data;
};


matrice* zeros(int lignes, int colonnes);
void print_mat(matrice* mat_device);
matrice* random_mat(int lignes, int colonnes, double x);
void dot_par(matrice* A,matrice* B,matrice* C);
void hadamar(matrice* A,matrice* B,matrice* C);
void sum(matrice* A,matrice* B,matrice* C);
void diff(matrice* A,matrice* B,matrice* C);

matrice* transpose(matrice* A);
matrice* copy(matrice* A);


void mat_RELU(matrice* A,matrice* C);
void mat_RELU_d(matrice* A,matrice* C);


void mat_sigmoid(matrice* A,matrice* C);
void mat_sigmoid_d(matrice* A,matrice* C);

void mat_SOFT_MAX(matrice* A,matrice* C);

//TODO faire
void mat_SOFT_MAX_d(matrice*A,matrice* C);


#endif