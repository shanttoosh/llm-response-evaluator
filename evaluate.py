#!/usr/bin/env python3
"""
LLM Evaluation Pipeline - Command Line Interface

Evaluate AI response quality for relevance, hallucination detection,
and performance metrics.

Usage:
    python evaluate.py --conversation <path> --context <path>
    python evaluate.py -c sample-chat-conversation-01.json -x sample_context_vectors-01.json
"""

import asyncio
import argparse
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_evaluation
from src.config import update_config


def print_banner():
    """Print the CLI banner."""
    print("=" * 60)
    print("  LLM Evaluation Pipeline v1.0")
    print("  Automated AI Response Quality Assessment")
    print("=" * 60)
    print()


def print_summary(result):
    """Print evaluation summary to console."""
    summary = result.summary
    
    print("\n" + "=" * 60)
    print("  EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 Chat ID: {result.chat_id}")
    print(f"👤 User ID: {result.user_id}")
    print(f"📝 Turns Evaluated: {result.turns_evaluated}")
    
    # Overall score with emoji
    overall = summary.get('overall_score', 0)
    if overall >= 0.9:
        emoji = "✅"
    elif overall >= 0.7:
        emoji = "👍"
    elif overall >= 0.5:
        emoji = "⚠️"
    else:
        emoji = "❌"
    
    print(f"\n{emoji} Overall Score: {overall:.1%}")
    
    # Category scores
    print("\n📈 Category Scores:")
    for category, score in summary.get('category_scores', {}).items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"   {category.capitalize():15} [{bar}] {score:.1%}")
    
    # Quality distribution
    print("\n📊 Quality Distribution:")
    dist = summary.get('quality_distribution', {})
    print(f"   ✅ Excellent: {dist.get('excellent', 0)}")
    print(f"   👍 Good: {dist.get('good', 0)}")
    print(f"   ⚠️  Acceptable: {dist.get('acceptable', 0)}")
    print(f"   ❌ Poor: {dist.get('poor', 0)}")
    
    # Issues summary
    issues_count = summary.get('issues_count', 0)
    if issues_count > 0:
        print(f"\n⚠️  Total Issues Found: {issues_count}")
    else:
        print(f"\n✅ No significant issues found")
    
    print("\n" + "=" * 60)


def print_detailed_results(result, show_all=False):
    """Print detailed per-turn results."""
    print("\n" + "=" * 60)
    print("  DETAILED TURN EVALUATIONS")
    print("=" * 60)
    
    for turn_eval in result.turn_evaluations:
        turn_num = turn_eval.get('turn_number', '?')
        overall = turn_eval.get('overall_score', 0)
        
        print(f"\n📌 Turn {turn_num} (Score: {overall:.1%})")
        print(f"   Query: {turn_eval.get('user_query', '')[:60]}...")
        print(f"   Response: {turn_eval.get('ai_response', '')[:80]}...")
        
        # Show ground truth if available
        if 'ground_truth_note' in turn_eval:
            print(f"   ⚠️  Ground Truth Note: {turn_eval['ground_truth_note']}")
        
        if show_all:
            for eval_result in turn_eval.get('evaluations', []):
                category = eval_result.get('category', 'unknown')
                score = eval_result.get('score', 0)
                issues = eval_result.get('issues', [])
                
                print(f"\n   [{category.upper()}] Score: {score:.1%}")
                if issues:
                    for issue in issues[:3]:
                        print(f"      ⚠️ {issue[:70]}...")


def save_results(result, output_path: str):
    """Save evaluation results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(result.to_json(indent=2))
    print(f"\n💾 Results saved to: {output_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Pipeline - Evaluate AI response quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate.py -c sample-chat-conversation-01.json -x sample_context_vectors-01.json
  python evaluate.py --conversation chat.json --context vectors.json --output results.json
  python evaluate.py -c chat.json -x vectors.json --use-llm --api-key sk-...
        """
    )
    
    parser.add_argument(
        '-c', '--conversation',
        required=True,
        help='Path to chat conversation JSON file'
    )
    
    parser.add_argument(
        '-x', '--context',
        required=True,
        help='Path to context vectors JSON file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output path for evaluation results JSON (default: evaluation_results.json)'
    )
    
    parser.add_argument(
        '--use-llm',
        action='store_true',
        help='Use LLM-as-a-Judge (requires OPENAI_API_KEY or --api-key)'
    )
    
    parser.add_argument(
        '--api-key',
        help='OpenAI API key for LLM evaluation'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='LLM model to use (default: gpt-4o-mini)'
    )
    
    parser.add_argument(
        '-d', '--detailed',
        action='store_true',
        help='Show detailed per-turn evaluation results'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress banner and progress output'
    )
    
    parser.add_argument(
        '--json-only',
        action='store_true',
        help='Output only JSON results (no console summary)'
    )
    
    args = parser.parse_args()
    
    # Print banner
    if not args.quiet and not args.json_only:
        print_banner()
    
    # Validate input files
    if not os.path.exists(args.conversation):
        print(f"Error: Conversation file not found: {args.conversation}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.context):
        print(f"Error: Context file not found: {args.context}", file=sys.stderr)
        sys.exit(1)
    
    # Configure LLM
    use_mock = not args.use_llm
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
        use_mock = False
    
    if args.use_llm and not os.environ.get('OPENAI_API_KEY') and not args.api_key:
        print("Warning: --use-llm specified but no API key provided. Using heuristic evaluation.", file=sys.stderr)
        use_mock = True
    
    # Update config
    if args.model:
        update_config(model=args.model)
    
    # Run evaluation
    if not args.quiet and not args.json_only:
        print(f"📂 Loading conversation: {args.conversation}")
        print(f"📂 Loading context: {args.context}")
        print(f"🔧 Mode: {'LLM-as-a-Judge' if not use_mock else 'Heuristic (no API key)'}")
        print()
    
    try:
        result = asyncio.run(run_evaluation(
            args.conversation,
            args.context,
            use_mock_llm=use_mock,
            verbose=not args.quiet and not args.json_only
        ))
    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Output results
    if args.json_only:
        print(result.to_json())
    else:
        print_summary(result)
        
        if args.detailed:
            print_detailed_results(result, show_all=True)
    
    # Save to file
    output_path = args.output or "evaluation_results.json"
    save_results(result, output_path)
    
    # Exit with appropriate code
    overall_score = result.summary.get('overall_score', 0)
    if overall_score < 0.5:
        sys.exit(2)  # Poor quality
    elif overall_score < 0.7:
        sys.exit(1)  # Acceptable but needs improvement
    else:
        sys.exit(0)  # Good quality


if __name__ == "__main__":
    main()
