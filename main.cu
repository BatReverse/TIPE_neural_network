// main.cu — Point d'entrée du programme. Tout le travail est délégué aux
// scénarios définis dans neural_network_tests.cu ; ce fichier ne fait que
// choisir lequel lancer.
#include"matrice.h"
#include<stdio.h>
#include"neural_network_tests.cu"

int main(int argc, char const *argv[])
{
    // Scénario actif : entraînement MNIST avec l'optimiseur Adam (voir
    // neural_network_tests.cu). Les autres lignes ci-dessous sont d'autres
    // scénarios disponibles ; décommenter celle voulue (une seule à la fois)
    // pour la lancer à la place.
    train_and_test_MNIST_opt();
    // train_and_test_MNIST_batch();  // entraînement par batch multi-threads
    // test_copy_neural_network();    // vérifie la copie profonde d'un réseau
    // train_and_test_MNIST();        // entraînement MNIST par SGD classique
    // test_perf();                   // mesure le temps de propagation avant
    return 0;
}
