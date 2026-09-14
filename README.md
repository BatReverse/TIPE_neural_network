# Réseau de neurones sur GPU (CUDA) — Projet TIPE

Implémentation "from scratch" d'un perceptron multicouche entraîné par
rétropropagation du gradient, avec toutes les opérations matricielles
exécutées sur GPU via CUDA. Le projet a été réalisé dans le cadre d'un TIPE
(Travail d'Initiative Personnelle Encadré) : **il n'a pas vocation à être un
code de production**, mais à explorer et illustrer le fonctionnement d'un
réseau de neurones et l'intérêt du calcul parallèle sur GPU.

L'application de démonstration entraîne un réseau à reconnaître les chiffres
manuscrits du jeu de données [MNIST](http://yann.lecun.com/exdb/mnist/).

## Prérequis

- Un GPU NVIDIA et le [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
  installé (`nvcc` doit être dans le PATH).
- Une bibliothèque pthreads (fournie nativement sous Linux).

> Le code ne peut ni compiler ni s'exécuter sans GPU NVIDIA : tous les
> calculs (matrices, activations, gradients...) sont faits sur GPU. Ce
> nettoyage du dépôt a donc été fait sans pouvoir recompiler/tester, faute de
> matériel compatible sur la machine utilisée — la logique du programme n'a
> volontairement pas été modifiée (voir plus bas la section "Ce qui n'a pas
> été touché").

## Compilation

```bash
./makemain.sh
```

Ce script appelle `nvcc` pour compiler tous les fichiers `.cu` du projet
(sauf `kernel.cu`, inclus directement par `matrice.cu`) et produit un
exécutable `main`.

## Utilisation

```bash
./main
```

Par défaut, `main()` (dans `main.cu`) lance `train_and_test_MNIST_opt()` :
entraînement indéfini (boucle infinie, à interrompre avec Ctrl+C) d'un
réseau 784 → 800 → 10 sur MNIST avec l'optimiseur Adam. Le réseau est
sauvegardé dans `mnistDNN.nn` au début de chaque génération, et le taux de
réussite ainsi que le MSE sont affichés sur la sortie standard après chaque
passage sur le jeu de test.

D'autres scénarios sont disponibles dans `neural_network_tests.cu` et
peuvent être activés en éditant `main.cu` (une seule ligne décommentée à la
fois) :

| Fonction | Description |
|---|---|
| `train_and_test_MNIST_opt()` | Entraînement MNIST avec l'optimiseur Adam (par défaut). |
| `train_and_test_MNIST_batch()` | Variante par batch, multi-threads (voir limitations ci-dessous). |
| `train_and_test_MNIST()` | Entraînement MNIST par descente de gradient classique (SGD), 2 générations. |
| `test_copy_neural_network()` | Vérifie la copie profonde d'un réseau. |
| `test_perf()` | Mesure le temps d'une propagation avant sur 5000 images. |

Le jeu de données MNIST est déjà présent dans `MNIST_dataset/` (fichiers au
format IDX, téléchargés depuis le site du LeCun Lab).

## Organisation du code

```
matrice.h / matrice.cu   Type `matrice` (GPU) et ses opérations (produit,
                         sommes, activations, transposition...)
kernel.h  / kernel.cu    Kernels CUDA de bas niveau utilisés par matrice.cu
                         et optimizer.cu
activation.h             Déclarations non utilisées (reliquat CPU, voir plus bas)

neural_network.h / .cu   Le réseau de neurones : création, propagation avant,
                         rétropropagation, sauvegarde/chargement, copie,
                         entraînement par batch multi-threads
optimizer.h / .cu        Optimiseur Adam (Momentum est géré directement dans
                         neural_network.cu)
Pile.h / Pile.cu         Pile chaînée de réseaux, utilisée comme pool de
                         copies pour l'entraînement par batch

MNIST_manager.h / .cu    Lecture du jeu de données MNIST (format IDX) et
                         conversion en matrices GPU
network_vis.cu           Utilitaire de debug : affiche un réseau en détail

neural_network_tests.cu  Scénarios d'entraînement/test (XOR, MNIST...)
main.cu                  Point d'entrée : choisit quel scénario lancer
makemain.sh              Script de compilation (nvcc)
```

Chaque fichier source contient désormais un commentaire d'en-tête expliquant
son rôle, et chaque fonction un commentaire résumant ce qu'elle fait — voir
directement le code pour le détail.

### Principe général

- Une `matrice` est une petite structure allouée côté CPU dont le champ
  `data` pointe vers un buffer alloué côté GPU (`cudaMalloc`). Toutes les
  opérations (`dot_par`, `sum`, `mat_sigmoid`, ...) lancent un kernel CUDA et
  écrivent leur résultat dans une matrice de sortie déjà allouée par
  l'appelant, pour limiter les allocations GPU répétées pendant
  l'entraînement.
- Un `neural_network` garde, pour chaque couche : sa pré-activation
  (`neuronnes_somme`), son activation (`neuronnes_activ`), ses poids/biais
  et les gradients correspondants (`dpoids`, `dbiais`, `dneuronnes`).
- La rétropropagation (`calcul_grad` dans `neural_network.cu`) calcule le
  delta de la dernière couche puis le propage vers les couches précédentes ;
  pour chaque couche, le produit `poids^T * delta_suivant` est calculé dans
  un thread pthread séparé pendant que le thread principal calcule la
  dérivée de l'activation, avant de combiner les deux résultats.
- Trois optimiseurs sont disponibles (`optimizer.h`) : `Rien` (SGD
  classique), `Momentum`, et `Adam`.

## Limitations connues et code non finalisé

Ce projet étant un TIPE et non un produit fini, certaines parties sont
volontairement incomplètes ou contiennent des bugs connus, laissés tels
quels lors de ce nettoyage pour ne pas changer le comportement du programme.
Ils sont signalés par des commentaires `ATTENTION` ou `TODO` dans le code :

- **Softmax non dérivable** : `mat_SOFT_MAX_d()` (dans `matrice.cu`) n'est
  pas implémentée. Ne pas utiliser `MIDLAYER`/`OUTPUTLAYER = 3` (softmax,
  voir `neural_network.h`) tant que ce n'est pas corrigé.
- **`mat_SOFT_MAX`** : la variable accumulant la somme des exponentielles
  n'est pas initialisée à 0 avant la boucle.
- **`mat_RELU_d`** (dans `matrice.cu`) appelle le kernel `cuda_RELU` au lieu
  de `cuda_RELU_d` : avec `MIDLAYER`/`OUTPUTLAYER = 1` (ReLU), la
  rétropropagation utilise `ReLU(x)` au lieu de sa dérivée `ReLU'(x)`.
- **Entraînement par batch** (`batch_training` dans `neural_network.cu`) :
  la moyenne du gradient sur le batch utilise une division entière
  (`1/batch_size`), qui vaut 0 dès que `batch_size > 1` — le gradient moyen
  est donc annulé au lieu d'être moyenné.
- **`activation.h`** déclare des fonctions d'activation scalaires (CPU)
  jamais définies ni appelées : reliquat d'une première version pensée pour
  tourner sans GPU. Les activations réellement utilisées sont les versions
  matricielles GPU (`mat_sigmoid`, `mat_RELU`, `mat_SOFT_MAX`).
- Le fichier `notes_pour_la_presentation.md` contient les notes de travail
  d'origine (observations sur Adam, pistes non terminées), conservées pour
  mémoire.

## Ce qui n'a pas été touché

À la demande explicite du propriétaire du projet, ce nettoyage s'est limité
à la forme (organisation, commentaires, suppression de code mort/`printf`
de debug oubliés, `.gitignore`) : **aucune logique de calcul, d'algorithme
ou de performance n'a été modifiée**. Les bugs listés ci-dessus sont donc
toujours présents et fonctionnent (ou pas) exactement comme avant.

## Nettoyage effectué sur le dépôt

- Ajout d'un `.gitignore` (binaire compilé, fichiers `.o`, fichiers
  temporaires d'éditeur).
- Le binaire compilé `main` a été retiré du suivi git (il reste généré
  localement par `makemain.sh`).
- Suppression des fichiers dupliqués `MNIST_dataset/train-images.idx3-ubyte`
  et `MNIST_dataset/train-labels.idx1-ubyte` (identiques aux fichiers
  `train-images-idx3-ubyte` / `train-labels-idx1-ubyte` réellement utilisés
  par le code).
- Suppression de quelques fonctions de test mortes/vides dans `main.cu`
  (`goofytest`, `neural_test`, jamais appelées) et de blocs de code commenté
  obsolètes.
- Remplacement de quelques messages de debug peu clairs (`"fdp"`,
  `"aaaaaa"`, `"ahhh"`) par des messages d'erreur explicites, sans changer le
  comportement du programme.
