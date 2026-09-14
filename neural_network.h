// neural_network.h — Réseau de neurones entièrement connecté (perceptron
// multicouche), entraîné par rétropropagation du gradient sur GPU. Toutes
// les matrices du réseau (poids, biais, activations...) vivent sur le GPU ;
// voir matrice.h pour la structure `matrice` et ses opérations.
#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK
#include"matrice.h"
#include"optimizer.h"
#include"MNIST_manager.h"

// Nombre maximum de threads lancés simultanément par batch_training()
// (entraînement par batch, voir Pile.h/neural_network.cu).
#define THREAD_MAX 16

//TODO faire tout

// Fonction d'activation utilisée pour les couches cachées (MIDLAYER) et pour
// la couche de sortie (OUTPUTLAYER) : 1 = RELU, 2 = sigmoïde, 3 = softmax.
// ATTENTION : mat_SOFT_MAX_d() n'est pas implémentée (voir matrice.cu), donc
// la valeur 3 casse la rétropropagation. Ne modifie que la config par défaut
// de tout le réseau : il n'y a pas d'activation différente par couche.
#define MIDLAYER 2
#define OUTPUTLAYER 2

// Un perceptron multicouche de `nombre_couche` couches (couche d'entrée
// comprise). Toutes les matrices ci-dessous sont indexées par couche.
struct neural_network
{
    int nombre_couche;
    // Nombre de neurones de chaque couche (taille nombre_couche).
    int* neuronnes_parcouche;

    // Pré-activation (poids*entrée + biais) de chaque couche.
    matrice** neuronnes_somme;
    // Activation (après fonction d'activation) de chaque couche ; l'entrée
    // du réseau est directement stockée dans neuronnes_activ[0].
    matrice** neuronnes_activ;

    // Poids/biais reliant la couche i à la couche i+1 (taille nombre_couche-1).
    matrice** poids;
    matrice** biais;

    // Gradients correspondants, recalculés à chaque rétropropagation.
    matrice** dpoids;
    matrice** dbiais;
    // Gradient de l'erreur par rapport à la pré-activation de chaque couche
    // (delta), utilisé en interne par calcul_grad().
    matrice** dneuronnes;

    // Taux d'apprentissage (learning rate) utilisé par les mises à jour.
    float vitesse_apprentissage;
};
typedef struct neural_network neural_network;

// Résultat d'une classification : indice du neurone de sortie le plus
// activé, et sa valeur d'activation.
struct result
{
    int indice;
    float valeur;
};
typedef struct result result;


// Alloue et initialise les matrices auxiliaires (moments...) nécessaires à
// l'optimiseur `type` (Rien/Momentum/Adam, voir optimizer.h) pour `reseau`.
optimizer* creer_optimizer(int type,neural_network* reseau,float Beta1,float Beta2);

// Applique une étape de mise à jour des poids/biais avec l'optimiseur `opt`,
// à partir des gradients dpoids/dbiais déjà calculés (calcul_grad).
void maj_reseau_opt(optimizer* opt,neural_network* reseau);

// Calcule le gradient (calcul_grad) puis met à jour le réseau avec `opt`.
void propagation_arriere_opt(optimizer* opt, neural_network* reseau,matrice* obj);


// Crée un réseau de `nb_couche` couches ; les arguments variadiques suivants
// donnent le nombre de neurones de chaque couche, ex :
// cree_reseau(3, 784, 800, 10) pour un réseau 784 -> 800 -> 10.
neural_network* cree_reseau(int nb_couche, ...);

// Propage `nourriture` (vecteur d'entrée) à travers le réseau ; le résultat
// est lisible dans reseau->neuronnes_activ[nombre_couche-1] après l'appel.
void propagation_avant(neural_network* reseau,matrice* nourriture);
// Calcule le gradient par rapport à la sortie attendue `obj` (calcul_grad)
// puis met à jour les poids/biais par descente de gradient classique (SGD).
void propagation_arriere(neural_network* reseau,matrice* obj);

// Renvoie l'indice (et la valeur) du neurone de sortie le plus activé,
// c.-à-d. la classe prédite par le réseau après une propagation avant.
result obtenir_resultat(neural_network* reseau);
// Coût quadratique (somme des carrés des écarts) entre la sortie actuelle
// du réseau et la sortie attendue `obj`.
float cout(neural_network* reseau, matrice* obj);
// Sauvegarde l'architecture et les poids/biais du réseau dans un fichier
// texte (format "maison", voir save_neural_network dans neural_network.cu).
void save_neural_network(neural_network* reseau,char* filename);
// Recharge un réseau depuis un fichier écrit par save_neural_network().
neural_network* importer(char* filename);
// Duplique entièrement un réseau (poids, biais, gradients...), utilisé pour
// l'entraînement par batch multi-threads (chaque thread a sa propre copie).
neural_network* copy_neural_network(neural_network* reseau);
// Libère toute la mémoire (CPU et GPU) associée au réseau.
void liberer_reseau(neural_network* reseau);

#endif