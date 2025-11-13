#!/usr/bin/env python3
"""
Complete Monitoring Demo - Training and Testing Documentation System

This demo shows the complete monitoring system including:
1. Detailed training step documentation with network serial IDs
2. Comprehensive testing phase monitoring with network outputs
3. Linkable data between training and testing phases
4. CSV and Excel output formats for analysis

Usage:
    python complete_monitoring_demo.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add src to path for development
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from cognitive_therapy_ai import (
    GameFactory, GameTrainer, OpponentFactory, 
    NetworkConfig, TrainingConfig, ExperimentConfig
)

def run_complete_monitoring_demo():
    """
    Demonstrate the complete training and testing monitoring system.
    """
    print("🎯 Complete Monitoring Demo - Training and Testing Documentation")
    print("=" * 70)
    
    # Create a minimal configuration for quick demo
    network_config = NetworkConfig(
        input_size=5,
        hidden_size=32,
        num_layers=1,
        dropout=0.1
    )
    
    training_config = TrainingConfig(
        learning_rate=0.01,
        num_epochs=3,  # Very short for demo
        batch_size=16,
        num_games_per_partner=10,  # Short sessions
        patience=5,
        convergence_threshold=0.001
    )
    
    experiment_config = ExperimentConfig(
        network_config=network_config,
        training_config=training_config,
        use_adaptive_loss=False,
        alpha=0.5
    )
    
    # Create demo directory
    demo_dir = Path("demo_complete_monitoring")
    demo_dir.mkdir(exist_ok=True)
    
    # Initialize components
    print("\n📊 Initializing Components...")
    game = GameFactory.create_game('prisoners-dilemma')
    trainer = GameTrainer(experiment_config)
    
    # Create opponents
    opponents = [
        OpponentFactory.create_opponent('probabilistic', defection_prob=0.3),
        OpponentFactory.create_opponent('probabilistic', defection_prob=0.7)
    ]
    
    print(f"✅ Game: {game.get_name()}")
    print(f"✅ Network Serial ID: {trainer.network_serial_id}")
    print(f"✅ Opponents: {len(opponents)} opponents")
    
    # PHASE 1: TRAINING WITH DETAILED MONITORING
    print("\n🔧 Phase 1: Training with Detailed Monitoring")
    print("-" * 50)
    
    # Train with automatic monitoring (save_dir triggers it)
    training_save_dir = demo_dir / "training_logs"
    results = trainer.train_multi_game(
        game_configs=[{
            'name': 'prisoners-dilemma',
            'weight': 1.0
        }],
        opponents=opponents,
        save_dir=str(training_save_dir)  # This enables training monitoring
    )
    
    print(f"✅ Training completed with {len(results)} epochs")
    
    # Check training monitoring files
    training_files = list(training_save_dir.glob("*.csv"))
    excel_files = list(training_save_dir.glob("*.xlsx"))
    
    print(f"✅ Training CSV files: {len(training_files)}")
    print(f"✅ Training Excel files: {len(excel_files)}")
    
    if training_files:
        print(f"   📁 Example training file: {training_files[0].name}")
    
    # PHASE 2: TESTING WITH DETAILED MONITORING
    print("\n🧪 Phase 2: Testing with Detailed Monitoring")
    print("-" * 50)
    
    # Test with detailed monitoring
    testing_save_dir = demo_dir / "testing_logs"
    
    # Single game evaluation with monitoring
    print("Running single game evaluation...")
    eval_results = trainer.evaluate(
        game=game,
        opponents=opponents,
        num_sessions=3,  # Short for demo
        enable_detailed_testing=True,
        testing_log_dir=str(testing_save_dir)
    )
    
    print(f"✅ Single evaluation completed")
    print(f"   Results for {len(eval_results)} opponent types")
    
    # Multi-game evaluation with monitoring
    print("Running multi-game evaluation...")
    games_dict = {
        'prisoners-dilemma': game,
        'stag-hunt': GameFactory.create_game('stag-hunt')
    }
    
    multi_eval_results = trainer.evaluate_on_multiple_games(
        games=games_dict,
        opponents=opponents[:1],  # Use fewer opponents for demo
        num_sessions=2,  # Short for demo
        enable_detailed_testing=True,
        testing_log_dir=str(testing_save_dir)
    )
    
    print(f"✅ Multi-game evaluation completed")
    print(f"   Results for {len(multi_eval_results)} games")
    
    # Check testing monitoring files
    testing_files = list(testing_save_dir.glob("*.csv"))
    testing_excel_files = list(testing_save_dir.glob("*.xlsx"))
    linkage_files = list(testing_save_dir.glob("*linkage*.csv"))
    
    print(f"✅ Testing CSV files: {len(testing_files)}")
    print(f"✅ Testing Excel files: {len(testing_excel_files)}")
    print(f"✅ Linkage files: {len(linkage_files)}")
    
    if testing_files:
        print(f"   📁 Example testing file: {testing_files[0].name}")
    if linkage_files:
        print(f"   🔗 Linkage file: {linkage_files[0].name}")
    
    # PHASE 3: ANALYSIS AND SUMMARY
    print("\n📈 Phase 3: Monitoring System Summary")
    print("-" * 50)
    
    print(f"🔑 Network Serial ID: {trainer.network_serial_id}")
    print("   This ID links all training and testing data for this network")
    
    print("\n📊 Training Monitoring Features:")
    print("   ✅ Step-by-step loss tracking (RL, opponent prediction, composite)")
    print("   ✅ Network head outputs (policy, opponent policy, value)")
    print("   ✅ Action sampling and rewards")
    print("   ✅ Network serial ID for data linkage")
    print("   ✅ Real-time CSV writing with Excel summaries")
    
    print("\n🧪 Testing Monitoring Features:")
    print("   ✅ Network prediction accuracy during evaluation")
    print("   ✅ Policy outputs and action distributions")
    print("   ✅ True vs predicted opponent behavior")
    print("   ✅ Reward tracking during testing")
    print("   ✅ Session-based organization")
    print("   ✅ Linkage files connecting training to testing data")
    
    print("\n📁 File Organization:")
    all_files = []
    all_files.extend(training_save_dir.glob("**/*") if training_save_dir.exists() else [])
    all_files.extend(testing_save_dir.glob("**/*") if testing_save_dir.exists() else [])
    
    csv_files = [f for f in all_files if f.suffix == '.csv']
    xlsx_files = [f for f in all_files if f.suffix == '.xlsx']
    
    print(f"   📄 Total CSV files: {len(csv_files)}")
    print(f"   📊 Total Excel files: {len(xlsx_files)}")
    print(f"   📂 Output directories: {demo_dir.name}/")
    print(f"      ├── training_logs/")
    print(f"      └── testing_logs/")
    
    # Show sample data if available
    if csv_files:
        print(f"\n🔍 Sample Data Preview:")
        sample_file = csv_files[0]
        try:
            import pandas as pd
            df = pd.read_csv(sample_file)
            print(f"   📄 File: {sample_file.name}")
            print(f"   📊 Rows: {len(df)}, Columns: {len(df.columns)}")
            print(f"   📋 Sample columns: {list(df.columns)[:5]}...")
            if 'network_serial_id' in df.columns:
                print(f"   🔑 Network Serial ID found: {df['network_serial_id'].iloc[0]}")
        except Exception as e:
            print(f"   ⚠️  Could not preview data: {e}")
    
    print("\n🎯 Demo Complete!")
    print("=" * 70)
    print("The complete monitoring system is now active and demonstrated.")
    print("All training and testing data is automatically logged and linked")
    print("using network serial IDs for comprehensive analysis.")
    print(f"\n📂 Check the '{demo_dir}' directory for all generated files.")
    
    return {
        'network_serial_id': trainer.network_serial_id,
        'training_results': results,
        'evaluation_results': eval_results,
        'multi_evaluation_results': multi_eval_results,
        'demo_directory': demo_dir
    }

if __name__ == "__main__":
    # Set up logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    try:
        results = run_complete_monitoring_demo()
        print(f"\n✅ Demo completed successfully!")
        print(f"Network ID: {results['network_serial_id']}")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)