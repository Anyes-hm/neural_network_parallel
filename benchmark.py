#!/usr/bin/env python3
"""
Benchmark Script for Parallel Neural Network
Runs multiple configurations and analyzes speedup
"""

import subprocess
import re
import os
import sys
from datetime import datetime

def run_experiment(num_processes, num_threads):
    """
    Run a single experiment with specified MPI processes and OpenMP threads
    """
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    
    try:
        result = subprocess.run(
            ['mpirun', '-np', str(num_processes), './neural_network'],
            capture_output=True,
            text=True,
            env=env,
            timeout=60
        )
        
        # Parse the output to extract timing
        output = result.stdout
        time_match = re.search(r'Total inference time: ([\d.]+) seconds', output)
        throughput_match = re.search(r'Throughput: ([\d.]+) samples/second', output)
        
        if time_match and throughput_match:
            return {
                'time': float(time_match.group(1)),
                'throughput': float(throughput_match.group(1)),
                'success': True
            }
        else:
            return {'success': False, 'error': 'Could not parse output'}
            
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': 'Timeout'}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    print("=" * 60)
    print("NEURAL NETWORK PARALLEL BENCHMARK")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check if executable exists
    if not os.path.exists('./neural_network'):
        print("Error: ./neural_network not found. Run 'make' first.")
        sys.exit(1)
    
    # Configuration matrix
    configs = [
        # (processes, threads_per_process, description)
        (1, 1, "Serial baseline"),
        (1, 2, "1 process, 2 threads"),
        (1, 4, "1 process, 4 threads"),
        (2, 1, "2 processes, 1 thread each"),
        (2, 2, "2 processes, 2 threads each"),
        (4, 1, "4 processes, 1 thread each"),
        (4, 2, "4 processes, 2 threads each"),
    ]
    
    results = []
    baseline_time = None
    
    print("Running experiments...")
    print("-" * 60)
    
    for processes, threads, description in configs:
        print(f"Testing: {description}...", end=" ", flush=True)
        
        result = run_experiment(processes, threads)
        
        if result['success']:
            time = result['time']
            throughput = result['throughput']
            
            # Calculate speedup vs baseline
            if baseline_time is None:
                baseline_time = time
                speedup = 1.0
            else:
                speedup = baseline_time / time
            
            efficiency = speedup / (processes * threads) * 100
            
            results.append({
                'config': description,
                'processes': processes,
                'threads': threads,
                'time': time,
                'throughput': throughput,
                'speedup': speedup,
                'efficiency': efficiency
            })
            
            print(f"✓ ({time:.4f}s, speedup: {speedup:.2f}x)")
        else:
            print(f"✗ ({result.get('error', 'Unknown error')})")
    
    # Print results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Configuration':<30} {'Time(s)':<10} {'Speedup':<10} {'Efficiency':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['config']:<30} {r['time']:<10.4f} {r['speedup']:<10.2f} {r['efficiency']:<10.1f}%")
    
    print()
    
    # Analysis
    if len(results) > 1:
        best_speedup = max(results, key=lambda x: x['speedup'])
        best_efficiency = max(results, key=lambda x: x['efficiency'])
        
        print("ANALYSIS:")
        print(f"  • Best speedup: {best_speedup['speedup']:.2f}x ({best_speedup['config']})")
        print(f"  • Best efficiency: {best_efficiency['efficiency']:.1f}% ({best_efficiency['config']})")
        print(f"  • Baseline time: {baseline_time:.4f}s")
        
        # Theoretical vs actual
        max_config = results[-1]
        theoretical_speedup = max_config['processes'] * max_config['threads']
        actual_speedup = max_config['speedup']
        parallel_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        print(f"\n  • Theoretical max speedup: {theoretical_speedup}x")
        print(f"  • Actual max speedup: {actual_speedup:.2f}x")
        print(f"  • Parallel efficiency: {parallel_efficiency:.1f}%")
        
        print("\nRECOMMENDATIONS:")
        if parallel_efficiency < 50:
            print("  ⚠ Low parallel efficiency. Consider:")
            print("    - Communication overhead is significant")
            print("    - Problem size may be too small for this parallelism")
            print("    - Load imbalance between processes")
        elif parallel_efficiency < 75:
            print("  ✓ Moderate parallel efficiency. Room for improvement:")
            print("    - Profile to identify bottlenecks")
            print("    - Optimize memory access patterns")
        else:
            print("  ✓✓ Excellent parallel efficiency!")
            print("    - Good balance between computation and communication")
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()
