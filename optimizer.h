// optimizer.h — Optimiseurs de descente de gradient : aucun (SGD classique,
// géré directement dans neural_network.cu par maj_reseau()), Momentum ou
// Adam. Un `optimizer` garde un jeu de matrices auxiliaires par couche
// (moments d'ordre 1 et 2, avec leur version "corrigée du biais" pour Adam).
#ifndef OPTIMIZER
#define OPTIMIZER
#include"matrice.h"

#define Rien 0
#define Momentum 1
#define Adam 2


struct optimizer{
    // Type d'optimiseur : Rien, Momentum ou Adam (voir #define ci-dessus).
    int type;
    // V : moment d'ordre 2 (moyenne mobile du carré du gradient) des poids.
    matrice** Vw;
    // Vc : V après correction de biais ("V chapeau").
    matrice** Vcw;
    // Idem pour les biais du réseau.
    matrice** Vb;
    matrice** Vcb;

    // M : moment d'ordre 1 (moyenne mobile du gradient), utilisé par Adam.
    matrice** Mw;
    matrice** Mcw;
    matrice** Mb;
    matrice** Mcb;

    float Beta1;
    float Beta2;
    float epsilon;

    // Beta1^t et Beta2^t (t = itération courante), utilisés pour la
    // correction de biais d'Adam ; mis à jour à chaque appel de apply_adam().
    float Beta1t;
    float Beta2t;

    // Numéro d'itération courant (utile pour Adam).
    int iteration;
};
typedef struct optimizer optimizer;

// Applique une étape d'Adam aux poids/biais de la couche `l` : met à jour
// les moments (Mw/Mb, Vw/Vb), leur version corrigée, puis les poids/biais
// eux-mêmes.
void apply_adam(optimizer* opt,matrice* W,matrice* biais,int l,matrice* dW,matrice* dBiais,float learning_rate);

#endif