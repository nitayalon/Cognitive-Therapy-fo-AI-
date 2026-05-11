"""
Master script to run all whole population analysis steps.
Runs data loading once, then generates all plots.
"""

import subprocess
import sys
from pathlib import Path

def run_script(script_name, description):
    """Run a Python script and report status."""
    print("\n" + "="*70)
    print(f"RUNNING: {description}")
    print("="*70)
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False
        )
        print(f"\n✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} failed with exit code {e.returncode}")
        return False

def main():
    print("="*70)
    print("WHOLE POPULATION ANALYSIS PIPELINE")
    print("="*70)
    print("\nThis script will run the following steps:")
    print("  1. Load and prepare data")
    print("  2. Extract TRUE cooperation rates from detailed logs")
    print("  3. Generate training loss plots")
    print("  4. Generate generalization heatmaps")
    print("  5. KLD analysis with TRUE cooperation rates")
    print("  6. KLD by test condition plots")
    print("  7. Test-time policy plots")
    print("  8. Agent generalization summary (avg reward & KLD)")
    print("  9. Export cooperation tables")
    
    # Step 1: Load data
    if not run_script('load_and_prepare_data.py', 'Data Loading & Preparation'):
        print("\nAborting pipeline due to data loading failure.")
        return
    
    # Step 2: Extract TRUE cooperation rates
    if not run_script('extract_true_cooperation_rates.py', 'Extract TRUE Cooperation Rates'):
        print("\nWarning: TRUE cooperation rate extraction failed. Continuing with other analyses...")
    
    # Step 3: Training plots
    run_script('plot_training_loss.py', 'Training Loss Plots')
    
    # Step 4: Generalization heatmaps
    run_script('plot_generalization_heatmaps.py', 'Generalization Heatmaps')
    
    # Step 5: KLD analysis
    run_script('plot_kld_analysis.py', 'KLD Analysis with TRUE Cooperation Rates')
    
    # Step 6: KLD by test condition
    run_script('plot_kld_by_test_condition.py', 'KLD by Test Condition Plots')
    
    # Step 7: Test-time policy plots
    run_script('plot_test_time_policies.py', 'Test-Time Policy Plots')
    
    # Step 8: Agent generalization summary
    run_script('plot_agent_generalization_summary.py', 'Agent Generalization Summary Plots')
    
    # Step 9: Export cooperation tables
    run_script('export_cooperation_table.py', 'Export Cooperation Tables')
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print("\nAll plots saved to:")
    print("  experiments/analysis_scripts/output/whole_population_generalization/figures/")
    print("\nProcessed data saved to:")
    print("  experiments/analysis_scripts/output/whole_population_generalization/data/")

if __name__ == '__main__':
    main()
