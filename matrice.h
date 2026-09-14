// matrice.h — Type "matrice" et opérations associées (produit, sommes,
// activations...) toutes exécutées sur le GPU via CUDA.
//
// Convention : une `matrice*` est une petite structure allouée sur le CPU
// (les champs lignes/colonnes) dont le champ `data` pointe vers un buffer
// alloué sur le GPU (cudaMalloc). Toutes les fonctions ci-dessous prennent
// donc des pointeurs "device" pour `data`, mais des pointeurs "host" pour
// les structures `matrice*` elles-mêmes.
//
// Sauf mention contraire, une fonction "C = f(A, B)" écrit son résultat dans
// une matrice `C` déjà allouée par l'appelant (aux bonnes dimensions) plutôt
// que d'en créer une nouvelle, pour éviter des allocations GPU répétées
// pendant l'entraînement.
#ifndef MATRICE
#define MATRICE
// Taille de bloc CUDA utilisée par défaut (Nl x Nl threads pour les kernels
// 2D comme le produit matriciel, Nl threads pour les kernels 1D).
#define Nl 32

struct matrice
{
    int lignes;
    int colonnes;
    float* data; // buffer GPU de taille lignes*colonnes (stocké en row-major)
};
typedef struct matrice matrice;

// Allocation ---------------------------------------------------------------

// Alloue une matrice sur le GPU initialisée à zéro.
matrice* zeros(int lignes, int colonnes);
// Affiche une matrice (copie GPU -> CPU puis printf), pour le débogage.
void print_mat(matrice* mat_device);
// Alloue une matrice sur le GPU et la remplit de valeurs aléatoires dans [-x, x].
matrice* random_mat(int lignes, int colonnes, float x);
// Libère une matrice (buffer GPU + structure CPU).
void free_mat(matrice* mat);


// Algèbre linéaire -----------------------------------------------------------

// Produit matriciel classique : C = A x B.
void dot_par(matrice* A,matrice* B,matrice* C);
// Produit terme à terme (Hadamard) : C = A .* B.
void hadamar(matrice* A,matrice* B,matrice* C);
// Somme terme à terme : C = A + B.
void sum(matrice* A,matrice* B,matrice* C);
// Différence terme à terme : C = A - B.
void diff(matrice* A,matrice* B,matrice* C);

// Renvoie une nouvelle matrice égale à la transposée de A.
matrice* transpose(matrice* A);
// Copie le contenu de A dans C (device to device, mêmes dimensions).
void copy(matrice* A,matrice* C);



// Fonctions d'activation (appliquées terme à terme) -------------------------

void mat_RELU(matrice* A,matrice* C);
void mat_RELU_d(matrice* A,matrice* C);


void mat_sigmoid(matrice* A,matrice* C);
void mat_sigmoid_d(matrice* A,matrice* C);

void mat_SOFT_MAX(matrice* A,matrice* C);

// TODO faire
// Dérivée du softmax : jamais implémentée (voir corps de la fonction).
// N'utilisez pas OUTPUTLAYER/MIDLAYER = 3 (softmax) pour la couche de
// sortie tant que cette fonction est vide, la rétropropagation serait fausse.
void mat_SOFT_MAX_d(matrice*A,matrice* C);

// TODO
// Dérivée du coût (MSE) par rapport à la sortie du réseau : C = 2*(A - obj).
void dCOST(matrice* A,matrice* obj, matrice* C);
// Multiplie A par un scalaire, en place : A *= lambda.
void multiply(matrice* A,float lambda);

// Différence "pondérée" utilisée pour la mise à jour des poids :
// C = A - alpha*B (alpha = vitesse d'apprentissage, ou 1 selon l'appelant).
void diff_avec_constante(matrice* A,matrice* B,matrice* C,float alpha);
// Met à jour une moyenne mobile (momentum/Adam) : V = beta*V + (1-beta)*W.
void update_momentum_velocity(matrice*V,matrice* W,float beta);

// Alloue une nouvelle matrice et copie le contenu de A dedans.
matrice* copy_new(matrice* A);
#endif
