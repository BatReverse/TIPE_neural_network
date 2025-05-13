#ifndef OPTIMIZER
#define OPTIMIZER
#include"matrice.h"

#define Rien 0
#define Momentum 1
#define Adam 2


struct optimizer{
    //rien, adam ou momentum 
    int type;
    // V pour les poids
    matrice** Vw;
    //V avec le chapeau
    matrice** Vcw;
    // V pour les biais
    matrice** Vb;
    //V avec le chapeau
    matrice** Vcb;
    
    matrice** Mw;
    matrice** Mcw;
    matrice** Mb;
    matrice** Mcb;

    float Beta1;
    float Beta2;
    float epsilon;


    float Beta1t;
    float Beta2t;

    //utile pour adam
    int iteration;
};
typedef struct optimizer optimizer;

void apply_adam(optimizer* opt,matrice* W,matrice* biais,int l,matrice* dW,matrice* dBiais,float learning_rate);

#endif