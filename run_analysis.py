import os
import subprocess
import time

def run_script(script_name, description):
    """Run a Python script and print its output"""
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    process = subprocess.Popen(['python', script_name], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    
    # Print output in real-time
    for line in process.stdout:
        print(line, end='')
    
    process.wait()
    elapsed_time = time.time() - start_time
    
    if process.returncode == 0:
        print(f"\n{'-'*80}")
        print(f"✅ {description} completed successfully in {elapsed_time:.2f} seconds.")
        print(f"{'-'*80}\n")
        return True
    else:
        print(f"\n{'-'*80}")
        print(f"❌ {description} failed with return code {process.returncode}.")
        print(f"{'-'*80}\n")
        return False

def main():
    """Run all analysis scripts in sequence"""
    print("Starting Service Program Impact Analysis Pipeline")
    print("="*80)
    
    # Create directories if they don't exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    # Step 1: Basic statistical analysis
    if not run_script('analyze_statistics.py', 'Basic Statistical Analysis'):
        print("Statistical analysis failed. Pipeline aborted.")
        return
    
    # Step 2: Create visualizations
    if not run_script('create_visualizations.py', 'Data Visualization'):
        print("Visualization creation failed. Continuing with regression analysis...")
    
    # Step 3: Regression analysis
    if not run_script('analyze_regression.py', 'Regression Analysis'):
        print("Regression analysis failed. Pipeline aborted.")
        return
    
    print("\n"+"="*80)
    print("Analysis pipeline completed successfully!")
    print("Results are available in the 'results' directory.")
    print("Visualizations are available in the 'figures' directory.")
    print("="*80)
    print("\nNow you can proceed with writing the scientific report based on these results.")

if __name__ == "__main__":
    main()
