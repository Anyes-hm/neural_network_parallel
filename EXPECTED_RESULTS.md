# Exemples de Résultats Attendus

Ce document montre des exemples de sorties attendues pour différentes configurations.

## 1. Exécution Standard (4 processus MPI)

```bash
$ make run
```

**Sortie Attendue :**
```
====================================
Parallel Neural Network Forward Pass
====================================
MPI Processes: 4
OpenMP Threads per process: 8
Total Samples: 1000
Samples per rank: 250
Network: 784 -> 128 -> 64 -> 10
====================================

Performance Results:
--------------------
Total inference time: 0.018234 seconds
Throughput: 54845.23 samples/second
Time per sample: 0.000182 seconds

Sample Predictions (first 3 samples):
-------------------------------------
Sample 0: [0.087, 0.104, 0.089, 0.112, 0.095, 0.098, 0.103, 0.091, 0.109, 0.102]
          Predicted class: 3 (confidence: 11.20%)

Sample 1: [0.093, 0.108, 0.096, 0.101, 0.089, 0.105, 0.097, 0.092, 0.111, 0.098]
          Predicted class: 8 (confidence: 11.10%)

Sample 2: [0.099, 0.095, 0.103, 0.097, 0.091, 0.102, 0.094, 0.106, 0.098, 0.115]
          Predicted class: 9 (confidence: 11.50%)
```

## 2. Version Séquentielle (Baseline)

```bash
$ ./neural_network_serial
```

**Sortie Attendue :**
```
====================================
Sequential Neural Network (Baseline)
====================================
Total Samples: 1000
Network: 784 -> 128 -> 64 -> 10

Performance Results:
--------------------
Total inference time: 0.125678 seconds
Throughput: 7956.78 samples/second
Time per sample: 0.001257 seconds

Sample Prediction (first sample):
----------------------------------
Sample 0: [0.087, 0.104, 0.089, 0.112, 0.095, 0.098, 0.103, 0.091, 0.109, 0.102]
```

## 3. Benchmark Complet

```bash
$ make benchmark
```

**Sortie Attendue :**
```
=== Benchmarking with different MPI configurations ===

>>> 1 process (serial baseline):
====================================
Parallel Neural Network Forward Pass
====================================
MPI Processes: 1
OpenMP Threads per process: 1
Total Samples: 1000
Samples per rank: 1000
Network: 784 -> 128 -> 64 -> 10
====================================

Performance Results:
--------------------
Total inference time: 0.128456 seconds
Throughput: 7784.23 samples/second
Time per sample: 0.001285 seconds

>>> 2 processes, 2 threads each:
====================================
Parallel Neural Network Forward Pass
====================================
MPI Processes: 2
OpenMP Threads per process: 2
Total Samples: 1000
Samples per rank: 500
Network: 784 -> 128 -> 64 -> 10
====================================

Performance Results:
--------------------
Total inference time: 0.037891 seconds
Throughput: 26391.45 samples/second
Time per sample: 0.000379 seconds

>>> 4 processes, 2 threads each:
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
Total inference time: 0.020567 seconds
Throughput: 48621.34 samples/second
Time per sample: 0.000206 seconds

>>> 4 processes, 4 threads each:
====================================
Parallel Neural Network Forward Pass
====================================
MPI Processes: 4
OpenMP Threads per process: 4
Total Samples: 1000
Samples per rank: 250
Network: 784 -> 128 -> 64 -> 10
====================================

Performance Results:
--------------------
Total inference time: 0.018234 seconds
Throughput: 54845.23 samples/second
Time per sample: 0.000182 seconds
```

## 4. Script Python Benchmark

```bash
$ python3 benchmark.py
```

**Sortie Attendue :**
```
============================================================
NEURAL NETWORK PARALLEL BENCHMARK
============================================================
Date: 2024-12-17 14:30:25

Running experiments...
------------------------------------------------------------
Testing: Serial baseline... ✓ (0.1285s, speedup: 1.00x)
Testing: 1 process, 2 threads... ✓ (0.0743s, speedup: 1.73x)
Testing: 1 process, 4 threads... ✓ (0.0441s, speedup: 2.91x)
Testing: 2 processes, 1 thread each... ✓ (0.0672s, speedup: 1.91x)
Testing: 2 processes, 2 threads each... ✓ (0.0379s, speedup: 3.39x)
Testing: 4 processes, 1 thread each... ✓ (0.0348s, speedup: 3.69x)
Testing: 4 processes, 2 threads each... ✓ (0.0206s, speedup: 6.24x)

============================================================
RESULTS SUMMARY
============================================================
Configuration                  Time(s)    Speedup    Efficiency
------------------------------------------------------------
Serial baseline                0.1285     1.00       100.0%    
1 process, 2 threads           0.0743     1.73       86.5%     
1 process, 4 threads           0.0441     2.91       72.8%     
2 processes, 1 thread each     0.0672     1.91       95.5%     
2 processes, 2 threads each    0.0379     3.39       84.8%     
4 processes, 1 thread each     0.0348     3.69       92.3%     
4 processes, 2 threads each    0.0206     6.24       78.0%     

ANALYSIS:
  • Best speedup: 6.24x (4 processes, 2 threads each)
  • Best efficiency: 95.5% (2 processes, 1 thread each)
  • Baseline time: 0.1285s

  • Theoretical max speedup: 8x
  • Actual max speedup: 6.24x
  • Parallel efficiency: 78.0%

RECOMMENDATIONS:
  ✓ Moderate parallel efficiency. Room for improvement:
    - Profile to identify bottlenecks
    - Optimize memory access patterns

============================================================
```

## 5. Comparaison Serial vs Parallel

```bash
$ make compare
```

**Sortie Attendue :**
```
=== Comparing Serial vs Parallel Performance ===

>>> Serial baseline:
====================================
Sequential Neural Network (Baseline)
====================================
Total Samples: 1000
Network: 784 -> 128 -> 64 -> 10

Performance Results:
--------------------
Total inference time: 0.125678 seconds
Throughput: 7956.78 samples/second
Time per sample: 0.001257 seconds

>>> Parallel (4 processes, 2 threads):
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
Total inference time: 0.020567 seconds
Throughput: 48621.34 samples/second
Time per sample: 0.000206 seconds

=== SPEEDUP: 6.11x ===
```

## 6. Analyse des Performances (Tableau Récapitulatif)

| Configuration       | Processus | Threads | Temps (s) | Débit (samples/s) | Speedup | Efficacité |
|--------------------|-----------|---------|-----------|-------------------|---------|------------|
| **Séquentiel**     | 1         | 1       | 0.1285    | 7,785            | 1.00x   | 100.0%     |
| Hybrid 1           | 1         | 2       | 0.0743    | 13,459           | 1.73x   | 86.5%      |
| Hybrid 2           | 1         | 4       | 0.0441    | 22,676           | 2.91x   | 72.8%      |
| MPI Only           | 2         | 1       | 0.0672    | 14,881           | 1.91x   | 95.5%      |
| Hybrid 3           | 2         | 2       | 0.0379    | 26,385           | 3.39x   | 84.8%      |
| MPI Only           | 4         | 1       | 0.0348    | 28,736           | 3.69x   | 92.3%      |
| **Optimal**        | 4         | 2       | 0.0206    | 48,543           | 6.24x   | 78.0%      |
| Max Threads        | 4         | 4       | 0.0182    | 54,945           | 7.06x   | 44.1%      |

**Observations :**
- ✅ Speedup quasi-linéaire jusqu'à 4-6 workers
- ⚠️ Efficacité décroît avec trop de threads (overhead, contention)
- ✅ Configuration optimale : 4 processus × 2 threads (~6x)
- ⚠️ Hyperthreading (16 threads sur 8 cores) = overhead > gain

## 7. Informations Système

```bash
$ make info
```

**Sortie Attendue :**
```
Compiler: mpicc
Flags: -O3 -fopenmp -Wall -Wextra -march=native
OpenMP support: 1
MPI version: mpirun (Open MPI) 4.1.2
```

## 8. Vérification des Warnings

```bash
$ make clean && make
```

**Sortie Idéale (aucun warning) :**
```
rm -f neural_network neural_network_serial *.o
mpicc -O3 -fopenmp -Wall -Wextra -march=native -o neural_network neural_network.c -lm -fopenmp
✓ Parallel version compiled!
gcc -O3 -Wall -o neural_network_serial neural_network_serial.c -lm
✓ Serial baseline compiled!
```

## Notes sur la Variance

**Variance Attendue :** ±5-10% entre exécutions
- Facteurs : charge système, cache, frequency scaling
- Atténuation : Plusieurs exécutions + médiane

**Exemple (3 runs) :**
```
Run 1: 0.0206s
Run 2: 0.0198s  ← Médiane
Run 3: 0.0211s
```

## Interprétation des Résultats

### Speedup Excellent (≥6x sur 8 workers)
- Parallélisation efficace
- Overhead minimal
- Bonne utilisation du matériel

### Speedup Bon (4-5x sur 8 workers)
- Parallélisation correcte
- Overhead modéré
- Amélioration possible

### Speedup Faible (<3x sur 8 workers)
- Problème de parallélisation
- Overhead excessif
- Revoir stratégie

## Validation Correctness

**Test de Reproductibilité :**
```bash
$ OMP_NUM_THREADS=1 mpirun -np 1 ./neural_network > output1.txt
$ OMP_NUM_THREADS=1 mpirun -np 1 ./neural_network > output2.txt
$ diff output1.txt output2.txt
# ✓ Aucune différence = déterministe
```

**Test de Cohérence Serial vs Parallel :**
```bash
$ ./neural_network_serial | grep "Sample 0" > serial.txt
$ OMP_NUM_THREADS=1 mpirun -np 1 ./neural_network | grep "Sample 0" > parallel.txt
$ diff serial.txt parallel.txt
# ✓ Mêmes prédictions (±10^-5 due to float precision)
```

---

**Note Importante :** Les temps exacts varieront selon votre matériel. L'important est :
1. Le **speedup relatif** (pas le temps absolu)
2. La **tendance** (croissance du speedup avec workers)
3. La **cohérence** des résultats entre exécutions
