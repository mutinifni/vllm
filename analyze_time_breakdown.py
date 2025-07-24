#!/usr/bin/env python3
"""
Analyze Mixtral and Qwen2-MoE timing CSV files with configurable iteration filtering.

Usage:
    python analyze_mixtral_timing.py timing.csv --skip-first 2 --skip-last 1
    python analyze_mixtral_timing.py timing.csv --steady-state-only

Supports both Mixtral and Qwen2-MoE timing files (auto-detected by CSV columns).
"""

import argparse
import pandas as pd
import sys
from pathlib import Path

def load_and_filter_data(csv_file, skip_first=0, skip_last=0, steady_state_only=False):
    """Load CSV data and filter iterations."""
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error loading {csv_file}: {e}")
        sys.exit(1)

    if df.empty:
        print("No data found in CSV file")
        sys.exit(1)

    print(f"Loaded {len(df)} total iterations")

    # Auto-detect steady state if requested
    if steady_state_only:
        # Skip first 25% and last 10% of iterations
        total_iters = len(df)
        skip_first = max(1, total_iters // 4)
        skip_last = max(1, total_iters // 10)
        print(f"Steady-state mode: skipping first {skip_first} and last {skip_last} iterations")

    # Apply filtering
    if skip_first > 0:
        df = df.iloc[skip_first:]
        print(f"Skipped first {skip_first} iterations")

    if skip_last > 0:
        df = df.iloc[:-skip_last]
        print(f"Skipped last {skip_last} iterations")

    if df.empty:
        print("No iterations remaining after filtering")
        sys.exit(1)

    print(f"Analyzing {len(df)} iterations (range: {df['iteration'].min()}-{df['iteration'].max()})")
    return df

def analyze_timing_data(df):
    """Analyze timing data and print statistics."""
    # Auto-detect model type based on columns
    if "shared_expert" in df.columns:
        model_name = "Qwen2-MoE"
        components = ["router", "experts", "shared_expert", "attention", "norm", "embeddings", "logits"]
    else:
        model_name = "Mixtral"
        components = ["router", "experts", "attention", "norm", "embeddings", "logits"]

    # Calculate statistics for each component
    stats = {}
    for component in components:
        if component in df.columns:
            values = df[component]
            stats[component] = {
                "mean": values.mean(),
                "min": values.min(),
                "max": values.max(),
                "std": values.std(),
                "count": len(values)
            }

    if not stats:
        print("No timing components found in data")
        return

    # Calculate total mean time per iteration
    total_mean = sum(stats[comp]["mean"] for comp in stats)

    # Print summary table
    print(f"\n=== {model_name} Timing Analysis ({len(df)} iterations) ===")
    print(f"{'Component':<12} {'Mean (ms)':<10} {'Std (ms)':<9} {'Min (ms)':<9} {'Max (ms)':<9} {'Count':<7} {'%':<6}")
    print("-" * 75)

    for component in components:
        if component in stats:
            s = stats[component]
            percentage = (s["mean"] / total_mean * 100) if total_mean > 0 else 0
            print(f"{component:<12} {s['mean']:<10.3f} {s['std']:<9.3f} {s['min']:<9.3f} {s['max']:<9.3f} {s['count']:<7} {percentage:<6.1f}%")

    print(f"{'TOTAL':<12} {total_mean:<10.3f}")
    print("=" * 75)

    # Additional insights
    print(f"\n=== Performance Insights ===")

    # Iteration timing analysis
    if len(df) > 1:
        iteration_totals = df[components].sum(axis=1)
        print(f"Per-iteration total time:")
        print(f"  Mean: {iteration_totals.mean():.3f} ms")
        print(f"  Std:  {iteration_totals.std():.3f} ms")
        print(f"  Min:  {iteration_totals.min():.3f} ms")
        print(f"  Max:  {iteration_totals.max():.3f} ms")

        # Coefficient of variation for stability
        cv = (iteration_totals.std() / iteration_totals.mean()) * 100
        print(f"  Coefficient of variation: {cv:.1f}% ({'stable' if cv < 10 else 'variable'})")

    # Component dominance
    print(f"\nComponent dominance:")
    sorted_components = sorted(stats.items(), key=lambda x: x[1]["mean"], reverse=True)
    for i, (comp, stat) in enumerate(sorted_components[:3]):
        percentage = (stat["mean"] / total_mean * 100)
        print(f"  {i+1}. {comp}: {percentage:.1f}% ({stat['mean']:.1f} ms)")

    # Model-specific efficiency ratios
    if "experts" in stats and "router" in stats:
        expert_router_ratio = stats["experts"]["mean"] / stats["router"]["mean"]
        print(f"\nMoE efficiency:")
        print(f"  Expert/Router ratio: {expert_router_ratio:.1f}x")
        print(f"  {'Efficient' if expert_router_ratio > 10 else 'Router overhead high'}")

        # Qwen2-MoE specific analysis
        if "shared_expert" in stats:
            shared_expert_ratio = stats["shared_expert"]["mean"] / stats["experts"]["mean"]
            print(f"  Shared Expert/Experts ratio: {shared_expert_ratio:.2f}x")
            total_moe_time = stats["experts"]["mean"] + stats["router"]["mean"] + stats["shared_expert"]["mean"]
            shared_expert_pct = (stats["shared_expert"]["mean"] / total_moe_time) * 100
            print(f"  Shared expert contribution: {shared_expert_pct:.1f}% of total MoE time")

    return stats

def save_detailed_analysis(df, output_file):
    """Save detailed per-iteration analysis to file."""
    # Auto-detect model type and components
    if "shared_expert" in df.columns:
        components = ["router", "experts", "shared_expert", "attention", "norm", "embeddings", "logits"]
    else:
        components = ["router", "experts", "attention", "norm", "embeddings", "logits"]

    # Add derived columns
    df_analysis = df.copy()
    df_analysis['total_time'] = df[components].sum(axis=1)

    # Model-specific derived columns
    if "shared_expert" in df.columns:
        # Qwen2-MoE specific columns
        df_analysis['moe_time'] = df[['router', 'experts', 'shared_expert']].sum(axis=1)
        df_analysis['vocab_time'] = df[['embeddings', 'logits']].sum(axis=1)
    else:
        # Mixtral specific columns
        df_analysis['moe_time'] = df[['router', 'experts']].sum(axis=1)
        df_analysis['vocab_time'] = df[['embeddings', 'logits']].sum(axis=1)

    # Add percentages
    for comp in components:
        df_analysis[f'{comp}_pct'] = (df[comp] / df_analysis['total_time'] * 100)

    df_analysis.to_csv(output_file, index=False, float_format='%.3f')
    print(f"\nDetailed analysis saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze Mixtral and Qwen2-MoE timing CSV files")
    parser.add_argument("csv_file", help="Path to timing CSV file")
    parser.add_argument("--skip-first", type=int, default=0,
                       help="Number of initial iterations to skip (default: 0)")
    parser.add_argument("--skip-last", type=int, default=0,
                       help="Number of final iterations to skip (default: 0)")
    parser.add_argument("--steady-state-only", action="store_true",
                       help="Automatically skip warmup/cooldown iterations")
    parser.add_argument("--output", type=str,
                       help="Save detailed analysis to CSV file")

    args = parser.parse_args()

    # Validate input file
    if not Path(args.csv_file).exists():
        print(f"Error: File {args.csv_file} not found")
        sys.exit(1)

    # Load and filter data
    df = load_and_filter_data(args.csv_file, args.skip_first, args.skip_last, args.steady_state_only)

    # Analyze data
    stats = analyze_timing_data(df)

    # Save detailed analysis if requested
    if args.output:
        save_detailed_analysis(df, args.output)

    print(f"\nSource: {args.csv_file}")

if __name__ == "__main__":
    main()