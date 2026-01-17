# Guide de Dépannage - Troubleshooting

Ce document recense les problèmes courants et leurs solutions.

---

## 🔧 Problèmes de Compilation

### Erreur : "mpicc: command not found"

**Symptôme :**
```bash
$ make
mpicc: command not found
```

**Solutions :**

**Ubuntu/Debian :**
```bash
sudo apt update
sudo apt install openmpi-bin libopenmpi-dev
```

**macOS :**
```bash
brew install open-mpi
```

**Vérification :**
```bash
which mpicc
mpicc --version
```

---

### Erreur : "omp.h: No such file or directory"

**Symptôme :**
```bash
neural_network.c:6:10: fatal error: omp.h: No such file or directory
 #include <omp.h>
```

**Solutions :**

**Ubuntu/Debian :**
```bash
sudo apt install libomp-dev
```

**macOS :**
```bash
brew install libomp
# Puis ajouter au Makefile :
# CFLAGS += -I/usr/local/opt/libomp/include
# LDFLAGS += -L/usr/local/opt/libomp/lib
```

**Vérification :**
```bash
echo | gcc -fopenmp -dM -E - | grep _OPENMP
# Doit afficher : #define _OPENMP 201511
```

---

### Erreur : "undefined reference to `MPI_Init'"

**Symptôme :**
```bash
/usr/bin/ld: /tmp/ccXXXXXX.o: undefined reference to `MPI_Init'
```

**Cause :** Utilisation de `gcc` au lieu de `mpicc`

**Solution :**
```bash
# Vérifier le Makefile
grep "^CC" Makefile
# Doit être : MPICC = mpicc

# Compiler avec mpicc
mpicc -fopenmp neural_network.c -o neural_network -lm
```

---

### Erreur : "undefined reference to `omp_get_max_threads'"

**Symptôme :**
```bash
undefined reference to `omp_get_max_threads'
```

**Cause :** Flag `-fopenmp` manquant

**Solution :**
```bash
# Ajouter -fopenmp à CFLAGS et LDFLAGS
make CFLAGS="-O3 -fopenmp -Wall" LDFLAGS="-lm -fopenmp"
```

---

## 🏃 Problèmes d'Exécution

### Erreur : "mpirun was unable to launch"

**Symptôme :**
```bash
$ make run
mpirun was unable to launch the specified application
```

**Solutions :**

1. **Vérifier que le programme existe :**
```bash
ls -l neural_network
# Si absent :
make clean && make
```

2. **Tester avec 1 processus :**
```bash
mpirun -np 1 ./neural_network
```

3. **Vérifier les permissions :**
```bash
chmod +x neural_network
```

---

### Erreur : "There are not enough slots available"

**Symptôme :**
```bash
There are not enough slots available in the system to satisfy the 8
slots that were requested
```

**Cause :** Demande de plus de processus que de cœurs disponibles

**Solutions :**

1. **Vérifier le nombre de cœurs :**
```bash
# Linux
nproc
# macOS
sysctl -n hw.ncpu
```

2. **Ajuster le nombre de processus :**
```bash
# Si 4 cœurs disponibles
mpirun -np 4 ./neural_network
```

3. **Forcer l'oversubscription (non recommandé pour benchmarks) :**
```bash
mpirun --oversubscribe -np 8 ./neural_network
```

---

### Erreur : Segmentation Fault

**Symptôme :**
```bash
$ ./neural_network
Segmentation fault (core dumped)
```

**Causes Possibles :**

1. **Stack overflow (allocation trop grande)**
```bash
# Augmenter la taille du stack
ulimit -s unlimited
```

2. **Déboguer avec gdb :**
```bash
# Compiler en mode debug
make CFLAGS="-g -fopenmp -O0"

# Lancer avec gdb
mpirun -np 1 gdb ./neural_network
(gdb) run
(gdb) backtrace
```

3. **Vérifier avec valgrind :**
```bash
mpirun -np 1 valgrind --leak-check=full ./neural_network
```

---

### Erreur : "Race condition detected"

**Symptôme :**
```bash
WARNING: ThreadSanitizer: data race
```

**Solution :**

1. **Compiler avec Thread Sanitizer :**
```bash
gcc -fsanitize=thread -fopenmp neural_network.c -o neural_network_test -lm
./neural_network_test
```

2. **Identifier la ligne problématique dans le rapport**

3. **Ajouter les bonnes clauses OpenMP :**
```c
// Mauvais
#pragma omp parallel for
for (int i = 0; i < n; i++) {
    result += array[i];  // Race condition!
}

// Bon
#pragma omp parallel for reduction(+:result)
for (int i = 0; i < n; i++) {
    result += array[i];
}
```

---

## 📊 Problèmes de Performance

### Speedup Faible (<2x avec 4 workers)

**Diagnostic :**

1. **Vérifier que le CPU n'est pas throttlé :**
```bash
# Linux
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
# Devrait être "performance", pas "powersave"

# Changer en mode performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

2. **Désactiver l'hyperthreading pour les benchmarks :**
```bash
echo off | sudo tee /sys/devices/system/cpu/smt/control
```

3. **Isoler les cœurs :**
```bash
# Utiliser taskset pour isoler
taskset -c 0-3 mpirun -np 4 ./neural_network
```

4. **Vérifier la charge système :**
```bash
top
# Autres processus consommant du CPU ?
```

---

### Efficacité Décroissante avec Plus de Threads

**Symptôme :**
```
2 threads: 1.8x speedup (90% efficiency)
4 threads: 2.9x speedup (72% efficiency)
8 threads: 4.1x speedup (51% efficiency)
```

**Explications :**

1. **Overhead de création/synchronisation de threads**
2. **Contention sur le cache L3**
3. **Bande passante mémoire saturée**

**Solutions :**

1. **Augmenter la taille du problème :**
```c
// Dans neural_network.c
#define TOTAL_SAMPLES 10000  // Au lieu de 1000
```

2. **Utiliser moins de threads par processus :**
```bash
# Au lieu de 4 procs × 4 threads
# Essayer 8 procs × 1 thread
OMP_NUM_THREADS=1 mpirun -np 8 ./neural_network
```

---

### Temps Variable entre Exécutions

**Symptôme :**
```
Run 1: 0.0234s
Run 2: 0.0456s  ← 2x plus lent !
Run 3: 0.0241s
```

**Causes :**

1. **Autres processus actifs**
2. **CPU frequency scaling**
3. **Cache froid vs chaud**

**Solutions :**

1. **Warmup avant mesure :**
```bash
# Première exécution pour "chauffer" le cache
./neural_network > /dev/null
# Puis mesurer
./neural_network
```

2. **Multiples exécutions + médiane :**
```bash
for i in {1..5}; do
    ./neural_network | grep "Total inference time"
done
```

3. **Isolation complète :**
```bash
# Désactiver turbo boost
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Fixer la fréquence CPU
sudo cpupower frequency-set -f 2.5GHz
```

---

## 🐛 Problèmes de Résultats

### Résultats Différents entre Exécutions

**Symptôme :**
```bash
$ ./neural_network | grep "Sample 0"
Sample 0: [0.087, 0.104, ...]

$ ./neural_network | grep "Sample 0"
Sample 0: [0.093, 0.098, ...]  # Différent !
```

**Cause :** Seed aléatoire différent

**Solution :**
```c
// Dans main()
srand(42);  // Seed fixe au lieu de time(NULL) + rank
```

---

### Prédictions Incohérentes (NaN, Inf)

**Symptôme :**
```
Sample 0: [nan, nan, nan, nan, ...]
```

**Causes Possibles :**

1. **Overflow dans softmax**
```c
// Vérifier l'implémentation softmax
// Doit soustraire max_val pour stabilité numérique
```

2. **Division par zéro**
```c
// Ajouter epsilon
output[i] = output[i] / (sum + 1e-8);
```

3. **Déboguer avec prints :**
```c
printf("DEBUG: max_val = %f, sum = %f\n", max_val, sum);
```

---

## 🔍 Outils de Diagnostic

### Profiling avec gprof

```bash
# Compiler avec profiling
mpicc -pg -fopenmp neural_network.c -o neural_network -lm

# Exécuter
mpirun -np 4 ./neural_network

# Analyser
gprof neural_network gmon.out > analysis.txt
less analysis.txt
```

### Profiling avec perf (Linux)

```bash
# Record
perf record mpirun -np 4 ./neural_network

# Report
perf report

# Annotate (voir le code source annoté)
perf annotate
```

### Memory Profiling avec Valgrind

```bash
# Détection de fuites mémoire
valgrind --leak-check=full --show-leak-kinds=all \
    mpirun -np 1 ./neural_network

# Cache profiling
valgrind --tool=cachegrind mpirun -np 1 ./neural_network
cg_annotate cachegrind.out.*
```

---

## 📝 Checklist de Validation

Avant de considérer le projet terminé :

- [ ] Compilation sans warnings (`make clean && make`)
- [ ] Exécution sans segfault (`make run`)
- [ ] Résultats déterministes (même seed → mêmes sorties)
- [ ] Speedup > 3x avec 4 processus × 2 threads
- [ ] Aucune race condition (Thread Sanitizer)
- [ ] Aucune fuite mémoire (Valgrind)
- [ ] Code documenté (commentaires Doxygen)
- [ ] Rapport complété (RAPPORT.md)
- [ ] Benchmarks exécutés (benchmark.py)

---

## 🆘 Obtenir de l'Aide

### Logs Détaillés

```bash
# MPI verbose
mpirun --mca btl_base_verbose 30 -np 4 ./neural_network

# OpenMP verbose
OMP_DISPLAY_ENV=TRUE ./neural_network
```

### Informations Système

```bash
# CPU info
lscpu

# MPI info
ompi_info

# OpenMP info
echo | gcc -fopenmp -E -dM - | grep -i openmp
```

### Forum et Documentation

- OpenMPI : https://www.open-mpi.org/community/help/
- OpenMP : https://www.openmp.org/resources/
- Stack Overflow : Tag [mpi] et [openmp]

---

## 📞 Contact Support

Si aucune solution ne fonctionne :

1. **Collecter les informations :**
```bash
./setup.sh > system_info.txt 2>&1
make clean && make > compile_log.txt 2>&1
make run > run_log.txt 2>&1
```

2. **Créer un rapport de bug** avec :
   - Version OS (`uname -a`)
   - Versions outils (`mpicc --version`, `gcc --version`)
   - Logs d'erreur complets
   - Étapes pour reproduire

3. **Contacter l'enseignant** avec ces informations

---

**Dernière mise à jour :** Décembre 2024
