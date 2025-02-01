#include<stdbool.h>
#include<stdio.h>
#include"matrice.h"
#include"inttypes.h"
#include"activation.h"
#include <stdlib.h>
#include <unistd.h>
#include"MNIST_manager.h"

void print_image(data_set* d,int k) {
    if (d == NULL || d->cur_data == NULL || d->cur_data->data->data == NULL) {
        printf("Aucune image disponible à afficher.\n");
        return;
    }

    int rows = d->lignes;
    int cols = d->colonnes;

    // printf("Label: %d\n", d->cur_data->label);
    // printf("Image:\n");

    // Parcours des pixels de l'image
    float* data_h = (float*)malloc(sizeof(float)* rows*cols);
    cudaMemcpy(data_h,d->cur_data[k].data->data,sizeof(float)* rows*cols,cudaMemcpyDeviceToHost);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            uint8_t pixel = data_h[i * cols + j]; // Accès au pixel
            // Choisir un caractère en fonction du niveau de gris
            if (pixel > 200) {
                printf("#");  // Pixel clair
            } else if (pixel > 150) {
                printf("o");
            } else if (pixel > 100) {
                printf("+");
            } else if (pixel > 50) {
                printf(".");
            } else {
                printf(" ");  // Pixel sombre
            }
        }
        printf("\n"); // Nouvelle ligne pour chaque rangée
    }
    printf("label: %d \n",d->cur_data[k].label);
}

void get_image_array(data_set* d){
    int n=d->lignes;
    int m=d->colonnes;
    int N= d->Nombre_image;
    data* c = (data*)malloc(sizeof(data)*N);
    for(int i=0;i<N;i++){
        c[i].data = zeros(n*m,1);
        c[i].label = fgetc(d->flabels);
        float* data_h = (float*)malloc(sizeof(float)*n*m);
        for(int j=0;j<n*m;j++){
            float t = fgetc(d->fimages);
            
            data_h[j] = t;
        }
        cudaMemcpy(c[i].data->data,data_h,sizeof(float)*n*m,cudaMemcpyHostToDevice);

        // free(data_h);
    }
    d->cur_data=c;
}

void get_next(data_set* d){
    if(!(d->fimages)){
        printf("ahhh");
        exit(EXIT_FAILURE);
    };
    int colonnes = d->cur_data->data->colonnes;
    int lignes =d->cur_data->data->lignes;
    float* data_h = (float*)malloc(sizeof(float)*lignes*colonnes);

    for(int i=0;i<colonnes*lignes;i++){
        data_h[i] = fgetc(d->fimages);
        // printf("%d",i);
    }
    (d->cur_data)->label =  fgetc(d->flabels);

    printf("%d",(d->cur_data)->label);
}

data_set* init(char* images_n,char* labels_n, bool is_training){
    FILE* images;
    data_set* res = (data_set*)malloc(sizeof(data_set));
    images = fopen(images_n,"r");

    if(images == NULL){
        printf("init failure");
        exit(EXIT_FAILURE);
    }


    int32_t magic_number = 0u;

    magic_number |= getc(images)<<24;
    magic_number |= getc(images)<<16; 
    magic_number |= getc(images)<<8;
    magic_number |= getc(images);

    int32_t number_of_image = 0u;

    number_of_image |= getc(images)<<24;
    number_of_image |= getc(images)<<16;
    number_of_image |= getc(images)<<8;
    number_of_image |= getc(images);

    int32_t rows = 0u;

    rows |= getc(images)<<24;
    rows |= getc(images)<<16;
    rows |= getc(images)<<8;
    rows |= getc(images);



    int32_t columns = 0u;

    columns |= getc(images)<<24;
    columns |= getc(images)<<16;
    columns |= getc(images)<<8;
    columns |= getc(images);

    res->fimages = images;

    


    // printf(" %d \n %d \n %d\n %d\n",magic_number,number_of_image,rows,columns);

    res->magic_number_image = magic_number;
    res->Nombre_image = number_of_image;
    res->lignes = rows;
    res->colonnes = columns;


  

    FILE* labels;
    labels = fopen(labels_n,"r");

    if(images == NULL){
        printf("init labels failure");
        exit(EXIT_FAILURE);
    }


    int32_t magic_number_labels = 0u;

    magic_number |= getc(labels)<<24;
    magic_number |= getc(labels)<<16; 
    magic_number |= getc(labels)<<8;
    magic_number |= getc(labels);

    //skip le nombre d'image
    getc(labels);
    getc(labels);
    getc(labels);
    getc(labels);
    res->magic_number_labels = magic_number_labels;
    res->is_training = is_training;

    res->flabels = labels;
    

    get_image_array(res);


    return res;
}

matrice** get_obj(int n){
    matrice** res = (matrice**)malloc(sizeof(matrice*)*n);
    for(int i=0;i<n;i++){
        res[i] = zeros(n,1);
        float* data_h = (float*)malloc(sizeof(float)*n);
        for(int j=0;j<n;j++){
            if(j==i){
                data_h[j] = 1.0;
            }
            else
                data_h[j] = 0.;
        }
        cudaMemcpy(res[i]->data,data_h,sizeof(float)*n,cudaMemcpyHostToDevice);
    }
    return res;
}

// int main(int argc, char const *argv[])
// {
//     data_set* a = init("MNIST_dataset/t10k-images-idx3-ubyte",
//         "MNIST_dataset/t10k-labels-idx1-ubyte",
//         true);

//     print_image(a,10);

//     return 0;
// }
