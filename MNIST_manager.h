#ifndef MNIST
#define MNIST

#include<stdbool.h>
#include<stdio.h>
#include"matrice.h"
#include"inttypes.h"
#include<stdlib.h>

struct data{
    uint8_t label;
    matrice* data;
};

typedef struct data data;

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


void get_next(data_set* d);
data_set* init(char* images_n,char* labels_n, bool is_training);
matrice** get_obj(int n);
void print_image(data_set* d,int k);
#endif