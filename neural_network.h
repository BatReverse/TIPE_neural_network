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

    double vitesse_apprentissage;
};
typedef struct neural_network neural_network;

struct result
{
    int indice;
    double valeur;
};
typedef struct result result;

neural_network* cree_reseau(int nb_couche, ...);

void propagation_avant(neural_network* reseau,matrice* nourriture);
void propagation_arriere(neural_network* reseau,matrice* obj);

result obtenir_resultat(neural_network* reseau);


#endif