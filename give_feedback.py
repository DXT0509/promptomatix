#!/usr/bin/env python3
"""
Script để give feedback cho optimized prompt và chạy optimization lại
Usage:
    python give_feedback.py <session_id> "Your feedback here"
    
Example:
    python give_feedback.py 1761015871.9096184 "The prompt should be more concise and focus on key points only"
"""

import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from promptomatix.main import save_feedback, optimize_with_feedback
from promptomatix.core.session import SessionManager

def main():
    if len(sys.argv) < 3:
        print("❌ Usage: python give_feedback.py <session_id> \"Your feedback here\"")
        print("\nExample:")
        print('  python give_feedback.py 1761015871.9096184 "Make the prompt shorter"')
        return 1
    
    session_id = sys.argv[1]
    feedback_text = sys.argv[2]
    
    print(f"📝 Session ID: {session_id}")
    print(f"💭 Your feedback: {feedback_text}")
    print()
    
    # Load session to get the optimized prompt
    session_manager = SessionManager()
    session = session_manager.get_session(session_id)
    
    if not session:
        print(f"❌ Session {session_id} not found!")
        print("Available sessions:")
        sessions_dir = os.path.join(os.path.dirname(__file__), 'sessions')
        if os.path.exists(sessions_dir):
            for f in sorted(os.listdir(sessions_dir))[-5:]:
                if f.endswith('.json'):
                    print(f"  - {f.replace('.json', '')}")
        return 1
    
    optimized_prompt = session.latest_optimized_prompt
    print(f"🎯 Current optimized prompt:\n{optimized_prompt}\n")
    
    # Save feedback
    try:
        save_feedback(
            text=optimized_prompt,
            start_offset=0,
            end_offset=len(optimized_prompt),
            feedback=feedback_text,
            prompt_id=session_id
        )
        
        # Ask if user wants to optimize with feedback now
        print("\n" + "="*60)
        response = input("🔄 Do you want to re-optimize with this feedback now? (y/n): ").strip().lower()
        
        if response == 'y' or response == 'yes':
            print("\n🚀 Starting optimization with your feedback...\n")
            result = optimize_with_feedback(session_id)
            
            if 'error' not in result:
                print("\n" + "="*60)
                print("✅ Optimization with feedback completed!")
                print("="*60)
                print(f"\n📊 Results:")
                print(f"Original Prompt:\n  {optimized_prompt}\n")
                print(f"New Optimized Prompt:\n  {result.get('result', 'N/A')}\n")
                if 'metrics' in result:
                    print(f"Cost: ${result['metrics'].get('cost', 0):.6f}")
                    print(f"Time: {result['metrics'].get('time_taken', 0):.2f}s")
            else:
                print(f"\n❌ Optimization failed: {result.get('error')}")
        else:
            print("\n✅ Feedback saved! You can run optimization later with:")
            print(f"   python -c \"from promptomatix.main import optimize_with_feedback; optimize_with_feedback('{session_id}')\"")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
