// MNIST_manager.h — Lecture du jeu de données MNIST au format IDX
// (http://yann.lecun.com/exdb/mnist/) et conversion en `matrice*` GPU
// utilisables directement par le réseau de neurones.
#ifndef MNIST
#define MNIST

#include<stdbool.h>
#include<stdio.h>
#include"matrice.h"
#include"inttypes.h"
#include<stdlib.h>

// Une image labellisée : `label` le chiffre (0-9), `data` l'image aplatie
// en vecteur colonne (28*28 = 784 lignes, 1 colonne), sur le GPU.
struct data{
    uint8_t label;
    matrice* data;
};

typedef struct data data;

// Un jeu de données MNIST entièrement chargé en mémoire GPU (cur_data
// contient les Nombre_image images), avec les métadonnées lues dans les
// en-têtes des fichiers IDX.
struct data_set{
    data* cur_data;
    bool is_training;
    int32_t magic_number_labels;
    int32_t magic_number_image;
    int32_t Nombre_image;
    int32_t lignes;
    int32_t colonnes;
    FILE* fimages;
    FILE* flabels;
};
typedef struct data_set data_set;


// Lit l'image/label suivant depuis les fichiers encore ouverts de `d`
// (usage ponctuel ; le chargement principal se fait via get_image_array,
// appelé automatiquement par init()).
void get_next(data_set* d);
// Ouvre les fichiers IDX d'images/labels, lit les en-têtes, puis charge tout
// le jeu de données sur le GPU. Termine le programme en cas d'échec d'ouverture.
data_set* init(char* images_n,char* labels_n, bool is_training);
// Construit les n vecteurs "one-hot" de taille n (n=10 pour les chiffres),
// utilisés comme sortie attendue lors de l'entraînement.
matrice** get_obj(int n);
// Affiche en ASCII (dans le terminal) la k-ième image du jeu de données `d`.
void print_image(data_set* d,int k);
#endif