# Guide d'Évaluation - Projet MPI/OpenMP Neural Network

## Vue d'Ensemble du Projet

Ce document guide l'évaluation du projet selon les critères académiques établis.

---

## 1. Correctness (Fonctionnement Correct)

### Tests à Effectuer

**Test 1 : Compilation Sans Erreurs**
```bash
make clean && make
# ✓ Doit compiler sans warnings ni erreurs
```

**Test 2 : Exécution Basique**
```bash
make run
# ✓ Doit produire des sorties cohérentes
# ✓ Pas de segfault, pas d'erreurs MPI
```

**Test 3 : Cohérence des Résultats**
```bash
# Exécuter plusieurs fois avec même seed
OMP_NUM_THREADS=1 mpirun -np 1 ./neural_network > output1.txt
OMP_NUM_THREADS=1 mpirun -np 1 ./neural_network > output2.txt
# ✓ Les prédictions doivent être identiques
```

**Test 4 : Validation des Probabilités**
- Les sorties softmax doivent sommer à 1.0 (±0.001)
- Toutes les probabilités doivent être entre [0, 1]

### Points à Vérifier
- [ ] Programme se compile correctement
- [ ] Exécution sans crash
- [ ] Résultats déterministes (même seed → mêmes résultats)
- [ ] Sorties softmax valides (somme = 1)
- [ ] Pas de valeurs NaN ou Inf

**Score : ___ / 20 points**

---

## 2. MPI Decomposition

### Aspects à Évaluer

**A. Configuration des Communicateurs**
```c
// Fichier : neural_network.c, lignes ~230
MPI_Init(&argc, &argv);
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```
- [ ] Initialisation/finalisation correcte de MPI
- [ ] Utilisation appropriée de MPI_COMM_WORLD
- [ ] Gestion des rangs et tailles

**B. Distribution des Données**
```c
// Lignes ~255-260
float *local_inputs = (float*)malloc(BATCH_SIZE * INPUT_SIZE * sizeof(float));
```
- [ ] Chaque processus traite un batch distinct
- [ ] Taille des batches équilibrée entre ranks
- [ ] Pas de redondance inutile de données

**C. Communications MPI**
```c
// Synchronisation : ligne ~273
MPI_Barrier(MPI_COMM_WORLD);

// Réduction : ligne ~287
MPI_Reduce(&compute_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, ...);

// Collecte : ligne ~295
MPI_Gather(local_outputs, BATCH_SIZE * OUTPUT_SIZE, MPI_FLOAT, ...);
```
- [ ] Utilisation judicieuse de MPI_Barrier (synchronisation)
- [ ] MPI_Reduce pour agréger les temps
- [ ] MPI_Gather pour collecter les résultats
- [ ] Pas de communications inutiles dans la boucle de calcul

**D. Équilibrage de Charge**
- [ ] Tous les ranks font le même travail (BATCH_SIZE identique)
- [ ] Pas de ranks inactifs pendant le calcul
- [ ] Distribution statique appropriée pour ce problème

**E. Prévention des Deadlocks**
- [ ] Utilisation d'appels collectifs (safer que send/recv)
- [ ] Ordre déterministe des communications
- [ ] Pas de dépendances circulaires

### Points à Vérifier
- [ ] Pas de ranks inactifs (tous calculent)
- [ ] Communications minimales et efficaces
- [ ] Échelle correctement avec le nombre de processus
- [ ] Code MPI propre et idiomatique

**Score : ___ / 20 points**

---

## 3. OpenMP Parallelism

### Aspects à Évaluer

**A. Zones Parallélisées**

**1. Multiplication Matrice-Vecteur (ligne ~135)**
```c
#pragma omp parallel for private(j) schedule(static)
for (i = 0; i < rows; i++) {
    float sum = 0.0f;  // Variable locale
    for (j = 0; j < cols; j++) {
        sum += matrix[i * cols + j] * vector[j];
    }
    result[i] = sum;  // Écriture thread-safe
}
```
- [ ] `private(j)` : Variable d'itération privée ✓
- [ ] `sum` : Automatiquement privée (locale) ✓
- [ ] `schedule(static)` : Approprié pour charge uniforme ✓
- [ ] Pas de conflit d'écriture (indices différents) ✓

**2. Activation ReLU (ligne ~152)**
```c
#pragma omp parallel for
for (i = 0; i < size; i++) {
    vector[i] += bias[i];
    if (activation == 1) {
        vector[i] = relu(vector[i]);
    }
}
```
- [ ] Opérations élément par élément (thread-safe) ✓
- [ ] Pas de dépendances entre itérations ✓

**B. Absence de Race Conditions**

**Test de Validation :**
```bash
# Compiler avec Thread Sanitizer
gcc -fsanitize=thread -fopenmp neural_network.c -o neural_network_test -lm
OMP_NUM_THREADS=4 ./neural_network_test
# ✓ Doit rapporter 0 race conditions
```

**Analyse Statique :**
- [ ] Toutes les variables partagées sont en lecture seule
- [ ] Variables modifiées sont privées ou à indices distincts
- [ ] Pas d'utilisation de variables globales modifiables

**C. Choix du Schedule**
- [ ] `schedule(static)` justifié (charge uniforme)
- [ ] Pas de `schedule(dynamic)` inutile (overhead)

**D. Gestion des Threads**
- [ ] Nombre de threads configurable via OMP_NUM_THREADS
- [ ] Pas de création/destruction répétée de threads (efficient)

### Points à Vérifier
- [ ] Parallélisation aux bons endroits (boucles coûteuses)
- [ ] Pas de race conditions (validation Thread Sanitizer)
- [ ] Clauses OpenMP correctes (private, shared, schedule)
- [ ] Efficacité : speedup mesurable avec threads

**Score : ___ / 20 points**

---

## 4. Performance et Speedup

### Métriques à Mesurer

**A. Exécution du Benchmark**
```bash
make benchmark
# Ou
python3 benchmark.py
```

**B. Speedup Attendu**

Pour un système 8-cœurs :

| Configuration | Speedup Théorique | Speedup Réel Attendu |
|---------------|-------------------|----------------------|
| 1×1 (baseline) | 1.0x | 1.0x |
| 1×2 | 2.0x | 1.7-1.9x (85-95%) |
| 1×4 | 4.0x | 2.8-3.5x (70-87%) |
| 2×2 | 4.0x | 3.2-3.7x (80-92%) |
| 4×2 | 8.0x | 5.5-6.5x (69-81%) |

**C. Formules Utilisées**
```
Speedup = T_serial / T_parallel
Efficiency = Speedup / (P × T) × 100%
```

**D. Analyse Attendue**
- [ ] Speedup croissant avec le nombre de workers
- [ ] Efficacité décroissante (overhead communication/sync)
- [ ] Comparaison avec loi d'Amdahl
- [ ] Identification des bottlenecks (bande passante, sync, ...)

### Points à Vérifier
- [ ] Speedup mesurable et significatif (>3x avec 8 workers)
- [ ] Résultats cohérents avec la théorie
- [ ] Graphiques ou tableaux clairs
- [ ] Explication des écarts vs théorique

**Score : ___ / 15 points**

---

## 5. Code Clarity & Structure

### Aspects à Évaluer

**A. Documentation**
```c
/**
 * @brief Matrix-vector multiplication with OpenMP parallelization
 * @param matrix Matrix [rows x cols]
 * @param vector Input vector [cols]
 * @param result Output vector [rows]
 * @param rows Number of rows
 * @param cols Number of columns
 */
void matmul_vec(...) { ... }
```
- [ ] Commentaires de fonction (Doxygen-style)
- [ ] Explication de la stratégie de parallélisation
- [ ] Documentation des structures de données

**B. Lisibilité**
- [ ] Nommage clair des variables (hidden1, hidden2, logits)
- [ ] Indentation cohérente
- [ ] Séparation compute/communication
- [ ] Pas de "magic numbers" (utilisation de #define)

**C. Structure**
- [ ] Organisation logique (init → compute → gather → report)
- [ ] Fonctions de taille raisonnable (<100 lignes)
- [ ] Séparation des préoccupations (réseau, MPI, perf)

**D. Qualité du Code**
```bash
# Compilation avec warnings stricts
gcc -Wall -Wextra -Wpedantic neural_network.c
# ✓ Doit compiler sans warnings
```

### Points à Vérifier
- [ ] Code bien commenté (compréhensible sans execution)
- [ ] Structure claire et modulaire
- [ ] Pas de code redondant ou mort
- [ ] Respect des conventions C

**Score : ___ / 10 points**

---

## 6. Expériences & Analyse

### Contenu du Rapport (RAPPORT.md)

**A. Protocole Expérimental**
- [ ] Description du système de test (CPU, RAM, OS)
- [ ] Configurations testées (matrice de tests)
- [ ] Méthodologie (nb exécutions, isolation CPU, ...)

**B. Résultats**
- [ ] Tableaux de performance clairs
- [ ] Graphiques (speedup, efficiency)
- [ ] Comparaison serial vs parallel

**C. Analyse**
- [ ] Interprétation des résultats
- [ ] Identification des bottlenecks
- [ ] Comparaison théorie vs pratique
- [ ] Discussion des limites

**D. Justifications**
- [ ] Choix d'architecture expliqués
- [ ] Stratégie de parallélisation justifiée
- [ ] Trade-offs discutés

### Points à Vérifier
- [ ] Expériences reproductibles (protocole clair)
- [ ] Résultats présentés proprement
- [ ] Analyse approfondie et critique
- [ ] Hypothèses testées et vérifiées

**Score : ___ / 10 points**

---

## 7. Conclusion & Perspectives

### Contenu Attendu

**A. Synthèse**
- [ ] Objectifs atteints vs non atteints
- [ ] Apprentissages clés
- [ ] Limitations rencontrées

**B. Perspectives Court Terme**
- [ ] Optimisations immédiates (tiling, SIMD, ...)
- [ ] Améliorations d'algorithme
- [ ] Extensions fonctionnelles (CNNs, ...)

**C. Perspectives Long Terme**
- [ ] Model parallelism
- [ ] Training distribué
- [ ] Application à problèmes réels

**D. Réflexion Critique**
- [ ] Pertinence de l'approche hybride
- [ ] Cas d'usage appropriés
- [ ] Alternatives considérées

### Points à Vérifier
- [ ] Conclusion structurée et complète
- [ ] Perspectives réalistes et pertinentes
- [ ] Réflexion critique sur le travail
- [ ] Ouverture vers recherche/industrie

**Score : ___ / 5 points**

---

## Grille Finale

| Critère | Points Max | Points Obtenus | Commentaires |
|---------|------------|----------------|--------------|
| 1. Correctness | 20 | | |
| 2. MPI Decomposition | 20 | | |
| 3. OpenMP Parallelism | 20 | | |
| 4. Performance & Speedup | 15 | | |
| 5. Code Clarity | 10 | | |
| 6. Expériences & Analyse | 10 | | |
| 7. Conclusion & Perspectives | 5 | | |
| **TOTAL** | **100** | | |

---

## Bonus Potentiels (+5 points max)

- [ ] Profiling avancé (gprof, perf, VTune)
- [ ] Visualisations graphiques des performances
- [ ] Implémentation d'optimisations avancées (SIMD, cache blocking)
- [ ] Extension à CNNs ou autres architectures
- [ ] Comparaison avec frameworks existants (TensorFlow, PyTorch)

---

## Critères de Qualité

### Excellent (90-100%)
- Tous les critères respectés
- Analyse approfondie et pertinente
- Code production-ready
- Perspectives innovantes

### Très Bien (80-89%)
- Critères majeurs respectés
- Bonne analyse des performances
- Code propre et fonctionnel
- Perspectives réalistes

### Bien (70-79%)
- Fonctionnement correct
- Analyse basique mais complète
- Code compréhensible
- Perspectives présentes

### Passable (60-69%)
- Fonctionnement partiel
- Analyse superficielle
- Code peu documenté
- Perspectives limitées

### Insuffisant (<60%)
- Ne compile pas ou crashe
- Pas d'analyse
- Code illisible
- Pas de perspectives

---

## Checklist Rapide pour l'Évaluateur

```bash
# 1. Compilation
cd neural_network_parallel
make clean && make

# 2. Test de base
make run

# 3. Benchmark
make benchmark

# 4. Vérification race conditions
# (Optionnel, nécessite recompilation)

# 5. Lecture du rapport
cat RAPPORT.md

# 6. Vérification du code
cat neural_network.c | grep -A 5 "pragma omp"
```

---

**Date d'évaluation :** _______________  
**Évaluateur :** _______________  
**Note finale :** _____ / 100
