#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include"matrice.h"
#include <stdarg.h>
//TODO faire tout

//1 RELU, 2 sigmoid, 3 softmax
#define MIDLAYER 2
#define OUTPUTLAYER 2
struct neural_network
{
    int nombre_couche;
    int* neuronnes_parcouche;

    matrice** neuronnes_somme;
    matrice** neuronnes_activ;

    matrice** poids;
    matrice** biais;

    matrice** dpoids;
    matrice** dbiais;
    matrice** dneuronnes;

    float vitesse_apprentissage;
};
typedef struct neural_network neural_network;

struct result
{
    int indice;
    float valeur;
};
typedef struct result result;

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

    //utile pour adam
    int iteration;
};
typedef struct  optimize optimzer;


optimizer* creer_optimizer(int type,neural_network* reseau,float Beta1,float Beta2);

void maj_reseau_opt(optimizer* opt,neural_network* reseau);

void propagation_arriere_opt(optimizer* opt, neural_network* reseau,matrice* obj);


neural_network* cree_reseau(int nb_couche, ...);

void propagation_avant(neural_network* reseau,matrice* nourriture);
void propagation_arriere(neural_network* reseau,matrice* obj);

result obtenir_resultat(neural_network* reseau);
float cout(neural_network* reseau, matrice* obj);
void save_neural_network(neural_network* reseau,char* filename);
neural_network* importer(char* filename);
#endif