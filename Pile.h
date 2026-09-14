// Pile.h — Pile chaînée de réseaux de neurones (`neural_network*`), utilisée
// par l'entraînement par batch multi-threads (train_and_test_MNIST_batch)
// comme pool de copies du réseau : chaque thread dépile une copie, calcule
// son gradient dessus, puis la rempile.
#ifndef PILE
#define PILE

#include"neural_network.h"
struct noeud{
    neural_network* val;
    struct noeud* next;
};
typedef struct noeud pile;

// Retire et renvoie le réseau en tête de pile (échoue par assert si vide).
neural_network* pop(pile** p);
// Empile `reseau` en tête de pile.
void empiler(pile** p,neural_network* reseau);
// Libère toute la pile ainsi que chaque réseau qu'elle contient.
void liberer_pile(pile* p);
// Recopie `poids`/`biais` dans chaque réseau de la pile (les remet à jour
// après une étape d'optimisation sur le réseau "maître").
void update_pile(pile* p, matrice** poids, matrice** biais);
// Entraîne un batch de `batch_size` exemples en parallèle (un thread par
// exemple, chacun utilisant une copie du réseau tirée de la pile `p`).
// Définie dans neural_network.cu (dépend de la structure neural_network).
void batch_training(int debut,int batch_size,data* nourriture,matrice** obj, int N,neural_network* reseau,pile* p);

#endif