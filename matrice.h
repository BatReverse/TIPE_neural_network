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

#endif