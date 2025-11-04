#!/usr/bin/env python3
"""
Test script để kiểm tra fix cho vấn đề score = 0.0
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from promptomatix.main import process_input

def test_optimization():
    """Test prompt optimization với một task đơn giản"""
    
    print("="*60)
    print("🧪 Testing Prompt Optimization Fix")
    print("="*60)
    
    result = process_input(
        raw_input="Answer Vietnamese cultural questions in 10 words or less",
        synthetic_data_size=3,
        train_ratio=0.33,
        task_type="generation",
        backend="simple_meta_prompt",
        model_name="meta-llama/llama-3.3-8b-instruct:free",
        model_api_key=os.environ.get("OPENROUTER_API_KEY"),
        model_api_base="https://openrouter.ai/api/v1",
        model_provider="openrouter",
        config_model_name="meta-llama/llama-4-maverick:free",
        config_model_api_key=os.environ.get("OPENROUTER_API_KEY"),
        config_model_api_base="https://openrouter.ai/api/v1",
        config_model_provider="openrouter",
        temperature=0.3,
        max_tokens=4000,
        config_max_tokens=14000
    )
    
    print("\n" + "="*60)
    print("📊 Test Results")
    print("="*60)
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return False
    
    metrics = result.get('metrics', {})
    initial_score = metrics.get('initial_prompt_score', 0.0)
    optimized_score = metrics.get('optimized_prompt_score', 0.0)
    
    print(f"Initial Score: {initial_score:.4f}")
    print(f"Optimized Score: {optimized_score:.4f}")
    
    if initial_score == 0.0 and optimized_score == 0.0:
        print("❌ FAIL: Both scores are still 0.0!")
        print("\nDEBUG INFO:")
        print(f"Backend: {result.get('backend')}")
        print(f"Session ID: {result.get('session_id')}")
        if 'synthetic_data' in result:
            print(f"Synthetic Data Count: {len(result.get('synthetic_data', []))}")
            if result.get('synthetic_data'):
                print(f"Sample Data: {result['synthetic_data'][0]}")
        return False
    else:
        print("✅ PASS: Scores are non-zero!")
        print(f"\nInitial Prompt:\n{result.get('initial_prompt', 'N/A')[:200]}...")
        print(f"\nOptimized Prompt:\n{result.get('result', 'N/A')[:200]}...")
        return True

if __name__ == "__main__":
    try:
        success = test_optimization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
