# TO-DO


- [ ] multi-batch training (date limite 15 mai)
  - [ ] fonction pour copier un reseau de neuronnes
  - [ ] fonction pour mettre a jour les poids
  - [ ] utiliser plusieurs thread (avec un semaphore ?)

- [ ] Etude des resultat
  - [ ] obtenir le graphe des des taux de reussite et du MSE en fonction de la géneration
  - [ ] etudier la vitesse de calcule pour chaque methode 
  - [ ] faire de meme avec les CNN
  - [ ] Comparer les resultats

# Notes

## Adam problemes rencontrer
Si adam est appliquer avec un learning rate eleve (0.1 ou 0.01) le modele reste bloque sur environ 10 % de winrate et converge des la premiere generation.
Cependant avec un learning rate de l'ordre de 0.001 il converge des la 3 eme generation a environ 92% de reussite

## Cas general
Le SGD ne semble jamais converger et reste entre certains pourcentage