# Parallel Neural Network - MPI/OpenMP Project

**Master 2 Intelligence Artificielle - Projet Académique**

## 📋 Description

Implémentation d'un forward pass parallèle pour un réseau de neurones MLP utilisant :
- **MPI** : Distribution des batches d'entrée entre processus
- **OpenMP** : Parallélisation des opérations matricielles

## 🎯 Architecture du Réseau

```
Input (784) → Hidden1 (128) → Hidden2 (64) → Output (10)
   [28×28]      [ReLU]          [ReLU]       [Softmax]
```

## 🚀 Installation Rapide

### Prérequis

**Ubuntu/Debian :**
```bash
sudo apt update
sudo apt install -y build-essential cmake
sudo apt install -y openmpi-bin openmpi-common libopenmpi-dev
sudo apt install -y libomp-dev python3
```

**macOS (avec Homebrew) :**
```bash
brew install gcc open-mpi libomp
```

**Windows (WSL2) :**
```bash
# Dans WSL Ubuntu
sudo apt update && sudo apt install -y build-essential openmpi-bin libopenmpi-dev
```

### Vérification de l'Installation

```bash
mpicc --version    # Doit afficher OpenMPI
gcc --version      # Doit supporter OpenMP
```

## 🔧 Compilation

```bash
# Compilation standard
make

# Compilation avec informations de debug
make CFLAGS="-g -fopenmp -Wall"

# Nettoyage
make clean
```

## ▶️ Exécution

### Exécution Simple

```bash
# Avec 4 processus MPI (configuration par défaut)
make run

# Ou directement
mpirun -np 4 ./neural_network
```

### Personnalisation du Nombre de Threads

```bash
# 4 processus MPI, 2 threads OpenMP par processus
OMP_NUM_THREADS=2 mpirun -np 4 ./neural_network

# 2 processus, 4 threads chacun
OMP_NUM_THREADS=4 mpirun -np 2 ./neural_network
```

### Benchmarking Automatique

```bash
# Lance plusieurs configurations et compare les performances
make benchmark

# Ou avec le script Python (plus détaillé)
python3 benchmark.py
```

## 📊 Exemple de Sortie

```
====================================
Parallel Neural Network Forward Pass
====================================
MPI Processes: 4
OpenMP Threads per process: 2
Total Samples: 1000
Samples per rank: 250
Network: 784 -> 128 -> 64 -> 10
====================================

Performance Results:
--------------------
Total inference time: 0.023456 seconds
Throughput: 42634.21 samples/second
Time per sample: 0.000234 seconds

Sample Predictions (first 3 samples):
-------------------------------------
Sample 0: [0.087, 0.104, 0.089, 0.112, 0.095, 0.098, 0.103, 0.091, 0.109, 0.102]
          Predicted class: 3 (confidence: 11.20%)
...
```

## 📈 Analyse des Performances

### Configuration Recommandée

Pour un système avec 8 cœurs :
```bash
# Option 1 : Maximiser le parallélisme MPI
OMP_NUM_THREADS=1 mpirun -np 8 ./neural_network

# Option 2 : Équilibré (généralement meilleur)
OMP_NUM_THREADS=2 mpirun -np 4 ./neural_network

# Option 3 : Favoriser OpenMP
OMP_NUM_THREADS=4 mpirun -np 2 ./neural_network
```

### Speedup Attendu

| Configuration | Speedup Attendu | Efficacité |
|--------------|----------------|------------|
| 1 proc × 1 thread | 1.0x (baseline) | 100% |
| 2 proc × 2 threads | 3.0-3.5x | 75-87% |
| 4 proc × 2 threads | 5.5-6.5x | 69-81% |

## 🧪 Tests et Validation

### Test de Correctness

```bash
# Compare résultats séquentiels vs parallèles
make test
```

### Profiling (Optionnel)

```bash
# Avec gprof
gcc -pg -fopenmp neural_network.c -o neural_network -lm -fopenmp
mpirun -np 4 ./neural_network
gprof neural_network gmon.out > analysis.txt

# Avec perf (Linux)
perf record mpirun -np 4 ./neural_network
perf report
```

## 📁 Structure du Projet

```
neural_network_parallel/
├── neural_network.c    # Code source principal
├── Makefile           # Compilation et exécution
├── benchmark.py       # Script de benchmarking avancé
├── RAPPORT.md         # Rapport académique complet
└── README.md          # Ce fichier
```

## 🐛 Dépannage

### Erreur : "mpicc: command not found"
```bash
# Vérifier l'installation MPI
which mpicc
# Si absent, réinstaller
sudo apt install openmpi-bin libopenmpi-dev
```

### Erreur : "undefined reference to omp_*"
```bash
# Vérifier le support OpenMP
echo |cpp -fopenmp -dM |grep -i open
# Ajouter -fopenmp aux flags de compilation
```

### Performance Médiocre
1. Vérifier que le CPU n'est pas throttlé :
   ```bash
   cpupower frequency-info  # Linux
   ```
2. Désactiver hyperthreading pour mesures :
   ```bash
   echo off | sudo tee /sys/devices/system/cpu/smt/control
   ```
3. Isoler les cœurs :
   ```bash
   taskset -c 0-7 mpirun -np 4 ./neural_network
   ```

## 📚 Ressources

- **Documentation MPI :** https://www.open-mpi.org/doc/
- **Documentation OpenMP :** https://www.openmp.org/specifications/
- **Tutoriels :** https://computing.llnl.gov/tutorials/

## 🎓 Évaluation Académique

Ce projet est évalué selon :
1. ✅ Correctness (fonctionnement correct)
2. ✅ MPI decomposition (pas de ranks inactifs, communications efficaces)
3. ✅ OpenMP parallelism (pas de race conditions)
4. ✅ Performance et speedup (analyse comparative)
5. ✅ Code clarity (commentaires, structure)
6. ✅ Expériences et analyse (benchmarks, justifications)
7. ✅ Conclusion (perspectives d'amélioration)

## 📝 Rapport

Le rapport académique complet est disponible dans `RAPPORT.md`. Il contient :
- Analyse théorique de la parallélisation
- Détails d'implémentation
- Résultats expérimentaux
- Perspectives d'amélioration

## 📧 Contact

Pour questions académiques, contactez votre enseignant.

## 📄 Licence

Projet académique - Master 2 IA

---

**Conseil :** Commencez par `make run` pour une exécution rapide, puis utilisez `benchmark.py` pour une analyse approfondie des performances.
