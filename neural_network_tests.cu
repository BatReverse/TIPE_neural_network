#include"neural_network.h"
#include"matrice.h"
#include<stdio.h>
#include"network_vis.cu"
#include"MNIST_manager.h"
#include<stdbool.h>
#include"inttypes.h"
#include<stdlib.h>
#include<unistd.h>

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


    cudaMalloc(&(un->data), sizeof(double) * 2);
    cudaMalloc(&(deux->data), sizeof(double) * 2);
    cudaMalloc(&(trois->data), sizeof(double) * 2);
    cudaMalloc(&(quatre->data), sizeof(double) * 2);
    
    double* un_h = (double*)malloc(sizeof(double)*2);
    double* deux_h = (double*)malloc(sizeof(double)*2);
    double* trois_h = (double*)malloc(sizeof(double)*2);
    double* quatre_h = (double*)malloc(sizeof(double)*2);

    un_h[0] = 0;
    un_h[1] = 0;

    deux_h[0] = 1;
    deux_h[1] = 1;

    trois_h[0] = 0;
    trois_h[1] = 1;

    quatre_h[0] = 1;
    quatre_h[1] = 0;

    cudaMemcpy(un->data,un_h,sizeof(double)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(deux->data,deux_h,sizeof(double)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(trois->data,trois_h,sizeof(double)*2,cudaMemcpyHostToDevice);
    cudaMemcpy(quatre->data,quatre_h,sizeof(double)*2,cudaMemcpyHostToDevice);

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

    cudaMalloc(&(zero->data), sizeof(double));
    cudaMalloc(&(un->data), sizeof(double));

    double* zero_h = (double*)malloc(sizeof(double));
    double* un_h = (double*)malloc(sizeof(double));

    zero_h[0] = 0; // Sortie XOR pour [0,0] et [1,1]
    un_h[0] = 1;   // Sortie XOR pour [0,1] et [1,0]

    cudaMemcpy(zero->data, zero_h, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(un->data, un_h, sizeof(double), cudaMemcpyHostToDevice);

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
            double sum=0;
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

    double* res_h = (double*)malloc(sizeof(double)*3);
    double* obj_h = (double*)malloc(sizeof(double)*3);

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

    cudaMalloc(&(res->data), sizeof(double)*3);
    cudaMalloc(&(obj->data), sizeof(double)*3);

    cudaMemcpy(res->data,res_h,sizeof(double)*3,cudaMemcpyHostToDevice);
    cudaMemcpy(obj->data,obj_h,sizeof(double)*3,cudaMemcpyHostToDevice);

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

double train_and_test_MNIST(){
    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    printf("set done\n");

    matrice** obj = get_obj(10);
    neural_network* reseau = cree_reseau(3,n,800,10);

    for (int i = 0; i < 5000; i++)
    {
        /* code */
        propagation_avant(reseau,t_train->cur_data[i].data);
        propagation_arriere(reseau,obj[t_train->cur_data[i].label]);
    }
    printf("done\n");
    
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

    print_image(t_labels,3);
}