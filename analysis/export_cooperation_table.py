"""
Export cooperation probability table for all test conditions.
Shows optimal policy and each agent's TRUE policy (from detailed logs) side-by-side.
"""

import pandas as pd
import numpy as np
from pathlib import Path

class CooperationProbabilityTableExporter:
    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def load_kld_data(self):
        """Load pre-computed KLD analysis with cooperation probabilities."""
        csv_path = self.data_dir / 'kld_analysis.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"KLD data not found at {csv_path}. Run plot_kld_analysis.py first.")
        
        print(f"Loading KLD data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"  Loaded {len(df)} KLD records")
        return df
    
    def create_cooperation_table(self, df):
        """
        Create table with columns:
        - Test Condition (game, opponent)
        - Optimal P(Cooperate)
        - PD Agent P(Cooperate)
        - SH Agent P(Cooperate)
        - HD Agent P(Cooperate)
        """
        print("\nCreating cooperation probability table...")
        
        # Pivot data to get one row per test condition
        # Group by test_game and opponent_prob
        test_conditions = df[['test_game', 'opponent_prob', 'optimal_coop_prob']].drop_duplicates()
        
        # Create readable condition labels
        game_names = {
            'prisoners-dilemma': 'Prisoners Dilemma',
            'stag-hunt': 'Stag Hunt',
            'hawk-dove': 'Hawk Dove'
        }
        
        opp_names = {
            0.1: 'Very Low (0.1)',
            0.3: 'Low (0.3)',
            0.5: 'Mid (0.5)',
            0.7: 'High (0.7)',
            0.9: 'Very High (0.9)'
        }
        
        # Build the table
        records = []
        
        # Sort by test game and opponent
        game_order = ['prisoners-dilemma', 'stag-hunt', 'hawk-dove']
        opp_order = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        for test_game in game_order:
            for opponent_prob in opp_order:
                # Get optimal cooperation prob
                optimal_row = df[(df['test_game'] == test_game) & 
                                (df['opponent_prob'] == opponent_prob)].iloc[0]
                optimal_coop = optimal_row['optimal_coop_prob']
                
                # Get each agent's cooperation probability
                pd_agent_coop = df[(df['train_game'] == 'prisoners-dilemma') & 
                                  (df['test_game'] == test_game) & 
                                  (df['opponent_prob'] == opponent_prob)]['agent_coop_prob'].values[0]
                
                sh_agent_coop = df[(df['train_game'] == 'stag-hunt') & 
                                  (df['test_game'] == test_game) & 
                                  (df['opponent_prob'] == opponent_prob)]['agent_coop_prob'].values[0]
                
                hd_agent_coop = df[(df['train_game'] == 'hawk-dove') & 
                                  (df['test_game'] == test_game) & 
                                  (df['opponent_prob'] == opponent_prob)]['agent_coop_prob'].values[0]
                
                records.append({
                    'Test Game': game_names[test_game],
                    'Opponent Defection Prob': opponent_prob,
                    'Opponent Type': opp_names[opponent_prob],
                    'Optimal P(Cooperate)': optimal_coop,
                    'PD Agent P(Cooperate)': pd_agent_coop,
                    'SH Agent P(Cooperate)': sh_agent_coop,
                    'HD Agent P(Cooperate)': hd_agent_coop
                })
        
        df_table = pd.DataFrame.from_records(records)
        
        return df_table
    
    def export_table(self, df_table):
        """Export table to CSV and display."""
        # Save to CSV
        csv_path = self.data_dir / 'cooperation_probabilities_by_condition.csv'
        df_table.to_csv(csv_path, index=False)
        print(f"\nSaved cooperation probability table to: {csv_path}")
        
        # Display the table
        print("\n" + "="*120)
        print("COOPERATION PROBABILITIES BY TEST CONDITION")
        print("="*120)
        
        # Format for display with aligned columns
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 120)
        pd.set_option('display.float_format', '{:.4f}'.format)
        
        print(df_table.to_string(index=False))
        print("="*120)
        
        # Also create a more compact version
        df_compact = df_table[['Test Game', 'Opponent Defection Prob', 
                               'Optimal P(Cooperate)', 'PD Agent P(Cooperate)', 
                               'SH Agent P(Cooperate)', 'HD Agent P(Cooperate)']].copy()
        
        csv_compact_path = self.data_dir / 'cooperation_probabilities_compact.csv'
        df_compact.to_csv(csv_compact_path, index=False, float_format='%.4f')
        print(f"\nAlso saved compact version to: {csv_compact_path}")
        
        # Summary statistics
        print("\n" + "="*120)
        print("SUMMARY STATISTICS")
        print("="*120)
        
        for agent in ['PD Agent', 'SH Agent', 'HD Agent']:
            col = f'{agent} P(Cooperate)'
            print(f"\n{agent}:")
            print(f"  Mean cooperation probability: {df_table[col].mean():.4f}")
            print(f"  Min cooperation probability:  {df_table[col].min():.4f}")
            print(f"  Max cooperation probability:  {df_table[col].max():.4f}")
            print(f"  Std cooperation probability:  {df_table[col].std():.4f}")
        
        print(f"\nOptimal Policy:")
        print(f"  Mean cooperation probability: {df_table['Optimal P(Cooperate)'].mean():.4f}")
        print(f"  Min cooperation probability:  {df_table['Optimal P(Cooperate)'].min():.4f}")
        print(f"  Max cooperation probability:  {df_table['Optimal P(Cooperate)'].max():.4f}")
        
        return df_table
    
    def export_deviation_table(self, df_table):
        """Export table showing deviation from optimal."""
        print("\n" + "="*120)
        print("DEVIATION FROM OPTIMAL (Agent - Optimal)")
        print("="*120)
        
        df_deviation = df_table[['Test Game', 'Opponent Defection Prob']].copy()
        
        for agent in ['PD Agent', 'SH Agent', 'HD Agent']:
            col = f'{agent} P(Cooperate)'
            df_deviation[f'{agent} Deviation'] = df_table[col] - df_table['Optimal P(Cooperate)']
        
        print(df_deviation.to_string(index=False))
        
        # Save
        csv_path = self.data_dir / 'cooperation_deviations_from_optimal.csv'
        df_deviation.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"\nSaved deviation table to: {csv_path}")
        
        return df_deviation
    
    def run(self):
        """Generate and export cooperation probability tables."""
        print("="*120)
        print("COOPERATION PROBABILITY TABLE EXPORTER")
        print("="*120)
        
        # Load data
        df = self.load_kld_data()
        
        # Create table
        df_table = self.create_cooperation_table(df)
        
        # Export
        self.export_table(df_table)
        
        # Export deviation table
        self.export_deviation_table(df_table)
        
        print("\n" + "="*120)
        print("EXPORT COMPLETE")
        print("="*120)


if __name__ == '__main__':
    exporter = CooperationProbabilityTableExporter(
        data_dir='experiments/analysis_scripts/output/whole_population_generalization/data',
        output_dir='experiments/analysis_scripts/output/whole_population_generalization'
    )
    exporter.run()
