import os
import pandas as pd
import matplotlib.pyplot as plt

# Read and plot results from each CSV in the current directory
def analyze_csv_results():
    for filename in os.listdir():
        if filename.endswith(".csv"):
            algo_name = filename.replace(".csv", "")
            print(f"Processing {filename}...")

            df = pd.read_csv(filename)

            # Plot 1: Serial vs Parallel runtimes
            plt.figure()
            plt.plot(df['N'], df['Serial(ms)'], marker='o', label='Serial')
            plt.plot(df['N'], df['Parallel(ms)'], marker='o', label='Parallel')
            plt.title(f"{algo_name} - Runtime")
            plt.xlabel("Input Size (N)")
            plt.ylabel("Time (ms)")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{algo_name}_runtime.png")

            # Plot 2: Speedup
            plt.figure()
            plt.plot(df['N'], df['Speedup'], marker='o', color='green')
            plt.title(f"{algo_name} - Speedup")
            plt.xlabel("Input Size (N)")
            plt.ylabel("Speedup (Serial / Parallel)")
            plt.xscale("log")
            plt.grid(True)
            plt.savefig(f"{algo_name}_speedup.png")

            print(f"Saved plots for {algo_name}.")

if __name__ == "__main__":
    analyze_csv_results()
