"""
Minimal test script to check if debugging works.
"""
import sys
import os

print("=== Debug Test Script ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Test basic functionality
print("\nTesting basic Python functionality...")
x = 5
y = 10
result = x + y
print(f"{x} + {y} = {result}")

print("\nTesting imports...")
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
except ImportError:
    print("✗ PyTorch not found")

try:
    import numpy as np
    print(f"✓ NumPy version: {np.__version__}")
except ImportError:
    print("✗ NumPy not found")

# Add src to path
print("\nAdding src to path...")
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)
print(f"Added path: {src_path}")

# Test our module imports
print("\nTesting cognitive therapy AI imports...")
try:
    from cognitive_therapy_ai.games import GameFactory
    print("✓ Games module imported successfully")
    
    game = GameFactory.create_game('prisoners-dilemma')
    print(f"✓ Game created: {game.name}")
    print(f"✓ State size: {game.get_state_size()}")
    
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Debug Test Complete ===")
print("If you see this message, debugging is working!")