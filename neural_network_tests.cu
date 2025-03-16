#include"neural_network.h"
#include"matrice.h"
#include<stdio.h>
#include"network_vis.cu"
#include"MNIST_manager.h"
#include<stdbool.h>
#include"inttypes.h"
#include<stdlib.h>
#include<unistd.h>

void afficher_tableau(int* tableau, int taille) {
    for (int i = 0; i < taille; i++) {
        printf("%d:%d", i, tableau[i]); // Affiche l'indice et la valeur
        if (i < taille - 1) {
            printf(", "); // Ajoute une virgule sauf pour le dernier élément
        }
    }
    printf("\n"); // Saut de ligne à la fin
}


matrice** creationressources_xor(){
    matrice** ressources = (matrice**)malloc(sizeof(matrice*)*4);

    matrice* un = (matrice*)malloc(sizeof(matrice));
    matrice* deux = (matrice*)malloc(sizeof(matrice));
    matrice* trois = (matrice*)malloc(sizeof(matrice));
    matrice* quatre = (matrice*)malloc(sizeof(matrice));

    un->lignes = 2;
    un->colonnes =1;

    deux->lignes = 2;
    deux->colonnes =1;
    
    trois->lignes = 2;
    trois->colonnes =1;
    
    quatre->lignes = 2;
    quatre->colonnes =1;


    cudaMalloc(&(un->data), sizeof(float) * 2);
    cudaMalloc(&(deux->data), sizeof(float) * 2);
    cudaMalloc(&(trois->data), sizeof(float) * 2);
    cudaMalloc(&(quatre->data), sizeof(float) * 2);
    
    float* un_h = (float*)malloc(sizeof(float)*2);
    float* deux_h = (float*)malloc(sizeof(float)*2);
    float* trois_h = (float*)malloc(sizeof(float)*2);
    float* quatre_h = (float*)malloc(sizeof(float)*2);

    un_h[0] = 0;
    un_h[1] = 0;

    deux_h[0] = 1;
    deux_h[1] = 1;

    trois_h[0] = 0;
    trois_h[1] = 1;

    quatre_h[0] = 1;
    quatre_h[1] = 0;

    cudaMemcpy(un->data,un_h,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(deux->data,deux_h,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(trois->data,trois_h,sizeof(float)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(quatre->data,quatre_h,sizeof(float)*2,cudaMemcpyHostToDevice);

    ressources[0] = un;
    ressources[1] = deux;
    ressources[2] = trois;
    ressources[3] = quatre;

    return ressources;
}


matrice** creationtests_xor() {
    matrice** tests = (matrice**)malloc(sizeof(matrice*) * 4);

    matrice* zero = (matrice*)malloc(sizeof(matrice));
    matrice* un = (matrice*)malloc(sizeof(matrice));

    zero->lignes = 1;
    zero->colonnes = 1;

    un->lignes = 1;
    un->colonnes = 1;

    cudaMalloc(&(zero->data), sizeof(float));
    cudaMalloc(&(un->data), sizeof(float));

    float* zero_h = (float*)malloc(sizeof(float));
    float* un_h = (float*)malloc(sizeof(float));

    zero_h[0] = 0; // Sortie XOR pour [0,0] et [1,1]
    un_h[0] = 1;   // Sortie XOR pour [0,1] et [1,0]

    cudaMemcpy(zero->data, zero_h, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(un->data, un_h, sizeof(float), cudaMemcpyHostToDevice);

    tests[0] = zero;  // Correspond à [0,0]
    tests[1] = zero;  // Correspond à [1,1]
    tests[2] = un;    // Correspond à [0,1]
    tests[3] = un;    // Correspond à [1,0]

    free(zero_h); // Libérer les données hôte
    free(un_h);

    return tests;
}


void test_xor(){
    neural_network* reseau = cree_reseau(3,2,4,1);
    matrice** ressources = creationressources_xor();
    matrice** test= creationtests_xor();

    for(int i=0;i<1000000;i++){
        int k = rand()%4;

        propagation_avant(reseau,ressources[k]);
        propagation_arriere(reseau,test[k]);
        if(i%10000 == 0){
            printf("i: %d\n",i);
            float sum=0;
            for(int i=0;i<4;i++){
                propagation_avant(reseau,ressources[i]);
                result r = obtenir_resultat(reseau);
                // print_mat(reseau->neuronnes_activ[0]);
                if(i == 0 || i== 1){
                    sum+=r.valeur*r.valeur;
                    printf("Output: %lf Expected: %lf \n",r.valeur,0.);
                }else{
                    sum+=(r.valeur-1)*(r.valeur-1);
                    printf("Output: %lf Expected: %lf \n",r.valeur,1.);
                    
                }
            }
            // print_mat(reseau->poids[reseau->nombre_couche-2]);
            // print_mat(reseau->dpoids[reseau->nombre_couche-2]);
            // print_mat(reseau->poids[0]);
            printf("MSE: %lf\n",sum/4.0);
            // print_mat(reseau->biais[2]);
        }
        // printf("i: %d \n",i);
    }
}

void testdCost(){

    float* res_h = (float*)malloc(sizeof(float)*3);
    float* obj_h = (float*)malloc(sizeof(float)*3);

    res_h[0] = 1;
    res_h[1] = 0.1;
    res_h[2] = 0.5;

    obj_h[0] = 0;
    obj_h[1] = 4;
    obj_h[2] = 0.4;

    matrice* res = (matrice*)malloc(sizeof(matrice));
    matrice* obj = (matrice*)malloc(sizeof(matrice));

    res->lignes = 3;
    res->colonnes=1;

    obj->lignes=3;
    obj->colonnes=1;

    cudaMalloc(&(res->data), sizeof(float)*3);
    cudaMalloc(&(obj->data), sizeof(float)*3);

    cudaMemcpy(res->data,res_h,sizeof(float)*3,cudaMemcpyHostToDevice);
    cudaMemcpy(obj->data,obj_h,sizeof(float)*3,cudaMemcpyHostToDevice);

    matrice* C = zeros(3,1);

    dCOST(res,obj,C);
    /* 
    supposer etre:
    2*(res-obj):
    2
    -7.8
    0.2
    */
    print_mat(C);
}

void testtranspose(){
    matrice* A = random_mat(5,1,10);
    print_mat(transpose(A));
    print_mat(A);
}

void melange_Fisher(data* data,int N){}

float train_and_test_MNIST(){
    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    int k=0;
    matrice** obj = get_obj(10);
    neural_network* reseau = importer("MNIST.nn");
    if(reseau==NULL){
        reseau = cree_reseau(4,n,800,800,10);
    }
    while (true)
    {
        save_neural_network(reseau,"MNIST.nn");
        int* wins = (int*)calloc(10,sizeof(int));
        for (int i = 0; i < t_train->Nombre_image; i++)
        {
            propagation_avant(reseau,t_train->cur_data[i].data);
            propagation_arriere(reseau,obj[t_train->cur_data[i].label]);
        }
        float sum = 0;
        int win =0;
        for (int i = 0; i < t_test->Nombre_image; i++)
        {
            propagation_avant(reseau,t_test->cur_data[i].data);
            int val= obtenir_resultat(reseau).indice == t_test->cur_data[i].label ? 1 : 0;
            win += val;
            wins[t_test->cur_data[i].label]+=val;
            sum+=cout(reseau,obj[t_test->cur_data[i].label]);   
        }
        printf("gen: %d\n",k);
        printf("MSE: %lf \n",sum/(float)t_test->Nombre_image);
        printf("winrate: %lf %\n",(float)win/((float)t_test->Nombre_image)*100.);
        afficher_tableau(wins,10);
        fflush(stdout);
        free(wins);
        k++;
    }
    return 0.;
}

void testdot(){
    matrice* A =random_mat(3,3,1);
    matrice* B =random_mat(3,3,1);
    matrice* C =random_mat(3,3,1);
    print_mat(A);
    print_mat(B);
    print_mat(C);
}

void testMNISTinit(){
    data_set* t_labels= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);

    print_image(t_labels,5);
}

void test_nnsave(){
    neural_network* reseau = cree_reseau(3,50,800,10);
    save_neural_network(reseau,"salut.nn");
    return;
}
void test_impnn(){
    neural_network* reseau = cree_reseau(3,50,800,10);
    save_neural_network(reseau,"salut.nn");
    reseau = importer("salut.nn");
    save_neural_network(reseau,"hey.nn");

}