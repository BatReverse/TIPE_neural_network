#ifndef PILE
#define PILE

#include"neural_network.h"
struct noeud{
    neural_network* val;
    struct noeud* next;
};
typedef struct noeud pile;

neural_network* pop(pile** p);
void empiler(pile** p,neural_network* reseau);
void liberer_pile(pile* p);
void update_pile(pile* p, matrice** poids, matrice** biais);
//dans neural_network.cu
void batch_training(int debut,int batch_size,data* nourriture,matrice** obj, int N,neural_network* reseau,pile* p);

#endif