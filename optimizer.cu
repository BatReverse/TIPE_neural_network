// optimizer.cu — Implémentation de l'optimiseur Adam (Kingma & Ba, 2014).
// Les autres optimiseurs (Momentum, SGD "Rien") n'ont pas besoin de code
// dédié ici : leurs mises à jour tiennent en un ou deux appels aux fonctions
// génériques de matrice.cu, faits directement dans maj_reseau_opt()
// (neural_network.cu).
#include"optimizer.h"
#include"kernel.h"
#include<stdio.h>




// Une étape d'Adam pour les poids/biais de la couche `l` :
//   1. met à jour les moments d'ordre 1 (M) et 2 (V) avec le gradient courant
//   2. calcule leur version corrigée du biais (Mc, Vc)
//   3. applique la mise à jour finale sur W et biais
// Beta1t/Beta2t (Beta^t) sont incrémentés ici, avant utilisation, car c'est
// la seule fonction qui avance le numéro d'itération de l'optimiseur.
void apply_adam(optimizer* opt,matrice* W,matrice* biais,int l,matrice* dW,matrice* dBiais,float learning_rate){
    dim3 blockDim(Nl);
    dim3 gridDimW((W->colonnes*W->lignes+blockDim.x - 1)/blockDim.x);
    dim3 gridDimB((biais->colonnes*biais->lignes+blockDim.x - 1)/blockDim.x);
    opt->Beta1t *= opt->Beta1;
    opt->Beta2t *= opt->Beta2;
    int Nw = W->colonnes*W->lignes;
    int Nb = biais->colonnes*biais->lignes;
    cudaError_t err = cudaGetLastError();
    cuda_adam_apply_momentum<<<gridDimW,blockDim>>>(opt->Mw[l]->data,opt->Beta1,dW->data,Nw);
    cuda_adam_apply_momentum<<<gridDimB,blockDim>>>(opt->Mb[l]->data,opt->Beta1,dBiais->data,Nb);

    cuda_adam_apply_speed<<<gridDimW,blockDim>>>(opt->Vw[l]->data,opt->Beta2,dW->data,Nw);
    cuda_adam_apply_speed<<<gridDimB,blockDim>>>(opt->Vb[l]->data,opt->Beta2,dBiais->data,Nb);


    // Vérifiez les erreurs CUDA après le lancement du kernel
    if (err != cudaSuccess) {
        printf("CUDA error after kernel launch adam: %s\n", cudaGetErrorString(err));
        return ;
    }

    // Synchronisation de l'appareil pour s'assurer que l'exécution a réussi


    cuda_adam_apply_hat<<<gridDimW,blockDim>>>(opt->Vcw[l]->data,opt->Vw[l]->data,opt->Beta2t,Nw);
    cuda_adam_apply_hat<<<gridDimW,blockDim>>>(opt->Mcw[l]->data,opt->Mw[l]->data,opt->Beta1t,Nw);


    cuda_adam_apply_hat<<<gridDimB,blockDim>>>(opt->Vcb[l]->data,opt->Vb[l]->data,opt->Beta2t,Nb);
    cuda_adam_apply_hat<<<gridDimB,blockDim>>>(opt->Mcb[l]->data,opt->Mb[l]->data,opt->Beta1t,Nb);

    cuda_adam_apply_weight<<<gridDimW,blockDim>>>(W->data,learning_rate,opt->Mcw[l]->data,opt->Vcw[l]->data,opt->epsilon,Nw);
    cuda_adam_apply_weight<<<gridDimB,blockDim>>>(biais->data,learning_rate,opt->Mcb[l]->data,opt->Vcb[l]->data,opt->epsilon,Nb);

    if (err != cudaSuccess) {
        printf("CUDA error after synchronization: %s\n", cudaGetErrorString(err));
        return ;
    }
}