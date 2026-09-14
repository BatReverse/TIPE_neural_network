// Pile.cu — Implémentation de la pile chaînée de réseaux (voir Pile.h).
#include"Pile.h"
#include"matrice.h"
#include"assert.h"

// Empile `reseau` en tête de pile.
void empiler(pile** p, neural_network* reseau) {
    pile* res = (pile*)malloc(sizeof(pile));
    res->val = reseau;
    res->next = *p;  // Lie à l'ancienne tête
    *p = res;        // Met à jour la tête de pile
}

// Retire et renvoie le réseau en tête de pile.
neural_network* pop(pile** p) {  // Passage par adresse pour modifier la pile
    assert(p != NULL && *p != NULL);  // Vérifie que la pile existe et n'est pas vide

    pile* tmp = *p;          // Récupère le premier élément
    neural_network* res = tmp->val;  // Sauvegarde la valeur à retourner
    *p = tmp->next;          // Met à jour la tête de pile

    return res;
}
// Libère toute la pile ainsi que chaque réseau qu'elle contient.
void liberer_pile(pile* p) {
    while (p != NULL) {
        pile* suivant = p->next;  
        if (p->val != NULL) {
            liberer_reseau(p->val);  
        }
        
        free(p);       
        p = suivant;   
    }
}

// Copie `poids`/`biais` dans les poids/biais d'un seul réseau (fonction
// interne à update_pile ; non déclarée dans Pile.h).
void update_reseau(neural_network* reseau,matrice** poids,matrice** biais){
    for (int i = 0; i < reseau->nombre_couche-1; i++)
    {
        copy(poids[i],reseau->poids[i]);
        copy(biais[i],reseau->biais[i]);
    }
}

// Recopie `poids`/`biais` dans chaque réseau de la pile `p`.
void update_pile(pile* p, matrice** poids, matrice** biais){
    pile* tmp = p;
    while (tmp != NULL)
    {
        update_reseau(tmp->val,poids,biais);
        tmp = tmp->next;
    }
    
}
