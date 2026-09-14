// neural_network_tests.cu — Scénarios de test/entraînement manuels : XOR
// (petit cas jouet pour valider la rétropropagation), et plusieurs variantes
// d'entraînement sur MNIST (SGD, Adam, par batch multi-threads). C'est ici
// que se trouve la logique "expérience" appelée depuis main.cu ; il n'y a
// pas de framework de test automatisé, chaque fonction s'exécute et affiche
// ses résultats sur stdout.
#include"neural_network.h"
#include"matrice.h"
#include<stdio.h>
#include"network_vis.cu"
#include"MNIST_manager.h"
#include<stdbool.h>
#include"inttypes.h"
#include"Pile.h"
#include<stdlib.h>
#include<unistd.h>

// Affiche "indice:valeur" pour chaque case du tableau, séparés par des
// virgules (utilisé pour afficher le nombre de bonnes réponses par chiffre).
void afficher_tableau(int* tableau, int taille) {
    for (int i = 0; i < taille; i++) {
        printf("%d:%d", i, tableau[i]); // Affiche l'indice et la valeur
        if (i < taille - 1) {
            printf(", "); // Ajoute une virgule sauf pour le dernier élément
        }
    }
    printf("\n"); // Saut de ligne à la fin
}


// Construit les 4 entrées possibles du XOR : (0,0), (1,1), (0,1), (1,0),
// chacune sous forme de vecteur colonne de taille 2 sur le GPU.
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


// Construit les sorties attendues correspondant à creationressources_xor() :
// XOR(0,0)=0, XOR(1,1)=0, XOR(0,1)=1, XOR(1,0)=1.
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

// Mesure grossièrement le temps de propagation avant sur 5000 images MNIST
// (pas d'entraînement : la ligne de rétropropagation est commentée), pour
// comparer le coût du forward seul à celui du forward+backward.
int test_perf(){
    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    matrice** obj = get_obj(10);
    neural_network* reseau = cree_reseau(3,n,800,10);

    // t_train->Nombre_image
    for (int i = 0; i <5000; i++)
    {
        propagation_avant(reseau,t_train->cur_data[i].data);
        // propagation_arriere(reseau,obj[t_train->cur_data[i].label]);
    }
    
    return 0.;
}

// Entraîne un petit réseau (2 -> 4 -> 1) sur le problème XOR par SGD
// classique : cas jouet qui sert à vérifier que la rétropropagation
// converge, avant de passer à MNIST. Affiche le MSE toutes les 10000
// itérations sur un total d'un million d'itérations.
void test_xor(){
    neural_network* reseau = cree_reseau(3,2,4,1);
    matrice** ressources = creationressources_xor();
    matrice** test= creationtests_xor();
    reseau->vitesse_apprentissage=0.01;
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

// Vérifie "à la main" le résultat de dCOST() sur un petit exemple numérique
// (voir le commentaire dans le corps pour le résultat attendu).
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

// Vérifie visuellement (print_mat) qu'une matrice aléatoire et sa
// transposée correspondent bien.
void testtranspose(){
    matrice* A = random_mat(5,1,10);
    print_mat(transpose(A));
    print_mat(A);
}

// Mélange aléatoirement (Fisher-Yates) le tableau d'exemples d'entraînement
// entre deux générations (époques), pour éviter que le réseau n'apprenne un
// ordre particulier.
void melange_Fisher(data* tab,int N){
    // Parcours du tableau de la fin au début
    for (int i = N-1; i > 0; i--)
    {
        int j = rand() % (i + 1);
        data temp = tab[i];
        tab[i] = tab[j];
        tab[j] = temp;
    }
    
}

// Entraîne un grand réseau (784 -> 2500 -> 2000 -> 1500 -> 1000 -> 500 -> 10)
// sur MNIST par SGD classique (propagation_arriere, sans optimiseur), sur 2
// générations (époques). Affiche à chaque génération le MSE, le taux de
// réussite global et par chiffre sur le jeu de test.
float train_and_test_MNIST(){
    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    matrice** obj = get_obj(10);
    neural_network* reseau = cree_reseau(7,n,2500,2000,1500,1000,500,10);
    
    int k=0;
    while (true)
    {
        int* wins = (int*)calloc(10,sizeof(int));
        // t_train->Nombre_image
        printf("debut\n");
        for (int i = 0; i <t_train->Nombre_image; i++)
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
        melange_Fisher(t_train->cur_data,t_train->Nombre_image);
        k++;
        if(k==2){
            break;
        }
    }
    return 0.;
}

// Alloue simplement trois matrices aléatoires et les affiche (ne teste pas
// réellement dot_par malgré son nom).
void testdot(){
    matrice* A =random_mat(3,3,1);
    matrice* B =random_mat(3,3,1);
    matrice* C =random_mat(3,3,1);
    print_mat(A);
    print_mat(B);
    print_mat(C);
}

// Vérifie que le chargement du jeu de test MNIST fonctionne en affichant
// l'image n°5 en ASCII.
void testMNISTinit(){
    data_set* t_labels= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);

    print_image(t_labels,5);
}

// Vérifie que save_neural_network() ne plante pas sur un petit réseau non entraîné.
void test_nnsave(){
    neural_network* reseau = cree_reseau(3,50,800,10);
    save_neural_network(reseau,"salut.nn");
    return;
}
// Vérifie l'aller-retour sauvegarde/chargement : sauvegarde un réseau,
// le recharge avec importer(), puis le resauvegarde pour comparaison manuelle.
void test_impnn(){
    neural_network* reseau = cree_reseau(3,50,800,10);
    save_neural_network(reseau,"salut.nn");
    reseau = importer("salut.nn");
    save_neural_network(reseau,"hey.nn");

}

// Entraîne un réseau 784 -> 800 -> 10 sur MNIST avec l'optimiseur Adam
// (learning rate 0.001), indéfiniment (boucle infinie). Sauvegarde le réseau
// dans "mnistDNN.nn" au début de chaque génération et affiche le MSE et le
// taux de réussite (global + par chiffre) sur le jeu de test. C'est la
// fonction appelée par défaut depuis main().
float train_and_test_MNIST_opt(){

    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    matrice** obj = get_obj(10);
    neural_network* reseau= cree_reseau(3,n,800,10);
    reseau->vitesse_apprentissage = 0.001;
    optimizer* opt = creer_optimizer(Adam,reseau,0.9,0.99);
    int k=0;
    while (true)
    {
        save_neural_network(reseau,"mnistDNN.nn");
        int* wins = (int*)calloc(10,sizeof(int));
        // t_train->Nombre_image
        printf("debut\n");
        for (int i = 0; i <t_train->Nombre_image; i++)
        {
            propagation_avant(reseau,t_train->cur_data[i].data);
            propagation_arriere_opt(opt,reseau,obj[t_train->cur_data[i].label]);
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
        melange_Fisher(t_train->cur_data,t_train->Nombre_image);
        k++;

    }
    return 0.;
}

// Variante par batch de l'entraînement MNIST : maintient une pile de
// THREAD_MAX copies du réseau, et traite les exemples par groupes de
// `batch_size` en parallèle (batch_training) avant chaque mise à jour Adam.
// Voir le commentaire sur batch_training() dans neural_network.cu pour un
// bug connu (division entière) affectant le gradient moyen du batch.
void train_and_test_MNIST_batch(){
    data_set* t_test= init("./MNIST_dataset/t10k-images-idx3-ubyte","./MNIST_dataset/t10k-labels-idx1-ubyte",true);
    data_set* t_train= init("./MNIST_dataset/train-images-idx3-ubyte","./MNIST_dataset/train-labels-idx1-ubyte",true);
    int n = t_test->colonnes*t_test->lignes;
    matrice** obj = get_obj(10);
    neural_network* reseau= cree_reseau(3,n,800,10);
    reseau->vitesse_apprentissage = 0.001;
    optimizer* opt = creer_optimizer(Adam,reseau,0.9,0.99);
    int batch_size = 100;
    int k=0;
    pile* p = NULL;
        printf("debut \n");

    for (int i = 0; i < THREAD_MAX; i++)
    {
        empiler(&p,copy_neural_network(reseau));
    }
    
    while (true)
    {
        int* wins = (int*)calloc(10,sizeof(int));
        // t_train->Nombre_image
        printf("debut \n");
        for (int i = 0; i <t_train->Nombre_image; i+=batch_size)
        {
            batch_training(i,batch_size,t_train->cur_data,obj,t_train->Nombre_image,reseau,p);
            maj_reseau_opt(opt,reseau);
            update_pile(p,reseau->poids,reseau->biais);
            printf("%d \n",i);
            // propagation_avant(reseau,t_train->cur_data[i].data);
            // propagation_arriere_opt(opt,reseau,obj[t_train->cur_data[i].label]);
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
        melange_Fisher(t_train->cur_data,t_train->Nombre_image);
        k++;
    }
}

// Vérifie que copy_neural_network() produit bien un réseau utilisable, en
// sauvegardant l'original puis la copie et en comparant les deux fichiers.
void test_copy_neural_network(){
    neural_network* reseau = cree_reseau(4,50,80,80,10);
    printf("aa\n");
    save_neural_network(reseau,"k.nn");
    printf("aa\n");

    reseau = copy_neural_network(reseau);

    printf("aa\n");
    save_neural_network(reseau,"kk.nn");
    printf("aa\n");

}