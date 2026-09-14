// network_vis.cu — Petit utilitaire de debug pour inspecter un réseau
// (dimensions, poids, biais, activations) directement dans le terminal.
// Non appelée par le flux d'entraînement principal ; utile pour vérifier
// "à la main" l'état d'un petit réseau (ex : le test XOR).
#include <stdio.h>
#include"neural_network.h"
#include"matrice.h"


// Affiche l'architecture d'un réseau ainsi que ses poids/biais/activations
// couche par couche (via print_mat, donc coûteux : à réserver au débogage).
void afficher_neural_network(neural_network* nn) {
    if (nn == NULL) {
        printf("Le réseau de neurones est NULL.\n");
        return;
    }

    printf("=== Détails du réseau de neurones ===\n");
    printf("Nombre de couches : %d\n", nn->nombre_couche);

    printf("Neurones par couche : ");
    for (int i = 0; i < nn->nombre_couche; i++) {
        printf("%d ", nn->neuronnes_parcouche[i]);
    }
    printf("\n");

    // Affichage des matrices pour chaque couche
    for (int i = 0; i < nn->nombre_couche - 1; i++) {
        printf("\n=== Couche %d -> Couche %d ===\n", i, i + 1);

        // Poids
        printf("Matrice des poids :\n");
        print_mat(nn->poids[i]);

        // Biais
        printf("Matrice des biais :\n");
        print_mat(nn->biais[i]);

        // Matrice somme des neurones
        printf("Neurones somme :\n");
        print_mat(nn->neuronnes_somme[i]);

        // Matrice activation des neurones
        printf("Neurones activés :\n");
        print_mat(nn->neuronnes_activ[i]);
    }

    // Affichage des paramètres supplémentaires
    printf("\n=== Paramètres supplémentaires ===\n");
    printf("Vitesse d'apprentissage : %f\n", nn->vitesse_apprentissage);

    printf("\n=== Fin de l'affichage ===\n");
}
