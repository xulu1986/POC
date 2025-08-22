#!/usr/bin/env python3
"""
Analyze and rank hyperparameter candidates from Bayesian optimization results.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

def load_optimization_results(filepath: str) -> Dict[str, Any]:
    """Load optimization results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_candidates(results: Dict[str, Any]) -> pd.DataFrame:
    """Convert optimization history to DataFrame and analyze."""
    
    # Extract optimization history
    history = results['optimization_history']
    
    # Convert to DataFrame
    rows = []
    for i, entry in enumerate(history):
        row = entry['params'].copy()
        row['score'] = entry['score']
        row['rank'] = i + 1  # Order in which they were evaluated
        row['timestamp'] = entry['timestamp']
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by score (descending - higher is better)
    df = df.sort_values('score', ascending=False).reset_index(drop=True)
    
    # Add performance ranking
    df['performance_rank'] = range(1, len(df) + 1)
    
    return df

def create_candidate_ranking(df: pd.DataFrame) -> None:
    """Create detailed ranking analysis."""
    
    print("=" * 80)
    print("🏆 HYPERPARAMETER CANDIDATE RANKING")
    print("=" * 80)
    
    print(f"\n📊 SUMMARY STATISTICS:")
    print(f"Total candidates evaluated: {len(df)}")
    print(f"Best score: {df['score'].max():.6f}")
    print(f"Worst score: {df['score'].min():.6f}")
    print(f"Average score: {df['score'].mean():.6f}")
    print(f"Score std deviation: {df['score'].std():.6f}")
    print(f"Score range: {df['score'].max() - df['score'].min():.6f}")
    
    # Identify top performers
    top_5 = df.head(5)
    
    print(f"\n🥇 TOP 5 CANDIDATES:")
    print("-" * 80)
    
    for idx, row in top_5.iterrows():
        print(f"\nRank #{row['performance_rank']} - Score: {row['score']:.6f}")
        print(f"  Batch Size: {int(row['batch_size']):,}")
        print(f"  Epochs: {int(row['epochs'])}")
        print(f"  Learning Rate: {row['learning_rate']:.6f}")
        print(f"  Embedding Dim: {int(row['embedding_dim'])}")
        # Only show dropout_rate and weight_decay if they exist
        if 'dropout_rate' in row:
            print(f"  Dropout Rate: {row['dropout_rate']:.4f}")
        if 'weight_decay' in row:
            print(f"  Weight Decay: {row['weight_decay']:.6f}")
    
    # Parameter analysis
    print(f"\n📈 PARAMETER ANALYSIS:")
    print("-" * 50)
    
    # Best performers (top 25%)
    top_25_percent = df.head(max(1, len(df) // 4))
    
    print(f"\nTop 25% performers ({len(top_25_percent)} candidates):")
    for param in ['batch_size', 'epochs', 'learning_rate', 'embedding_dim']:
        if param in df.columns:
            if param in ['batch_size', 'epochs', 'embedding_dim']:
                values = top_25_percent[param].astype(int)
                print(f"  {param}: {values.min()} - {values.max()} (avg: {values.mean():.0f})")
            else:
                values = top_25_percent[param]
                print(f"  {param}: {values.min():.6f} - {values.max():.6f} (avg: {values.mean():.6f})")
        else:
            print(f"  {param}: N/A (not in results)")
    
    # Parameter correlations
    print(f"\n🔗 PARAMETER CORRELATIONS WITH PERFORMANCE:")
    print("-" * 50)
    
    numeric_cols = ['batch_size', 'epochs', 'learning_rate', 'embedding_dim']
    correlations = []
    
    for param in numeric_cols:
        if param in df.columns:
            corr = df[param].corr(df['score'])
            correlations.append((param, corr))
            print(f"  {param}: {corr:+.3f}")
        else:
            print(f"  {param}: N/A (not in results)")
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"\n📊 MOST INFLUENTIAL PARAMETERS (by correlation):")
    print("-" * 50)
    for i, (param, corr) in enumerate(correlations[:3]):
        direction = "↑ Higher is better" if corr > 0 else "↓ Lower is better"
        print(f"  {i+1}. {param}: {corr:+.3f} {direction}")

def create_detailed_comparison(df: pd.DataFrame) -> None:
    """Create detailed comparison of top candidates."""
    
    print(f"\n" + "=" * 80)
    print("🔍 DETAILED CANDIDATE COMPARISON")
    print("=" * 80)
    
    # Compare top 3 with your historical best
    top_3 = df.head(3)
    
    # Historical best from your data
    historical_best = {
        'batch_size': 15000,
        'epochs': 30, 
        'learning_rate': 0.01,
        'embedding_dim': 96,
        'score': 0.2179
    }
    
    print(f"\n📋 TOP 3 CANDIDATES vs HISTORICAL BEST:")
    print("-" * 80)
    
    print(f"{'Metric':<15} {'Historical':<12} {'Rank #1':<12} {'Rank #2':<12} {'Rank #3':<12}")
    print("-" * 80)
    
    metrics = ['score', 'batch_size', 'epochs', 'learning_rate', 'embedding_dim']
    
    for metric in metrics:
        if metric in historical_best:
            hist_val = historical_best[metric]
        else:
            hist_val = "N/A"
        
        if metric in top_3.columns:
            if metric in ['batch_size', 'epochs', 'embedding_dim']:
                row = f"{metric:<15} {str(hist_val):<12} {int(top_3.iloc[0][metric]):<12} {int(top_3.iloc[1][metric]):<12} {int(top_3.iloc[2][metric]):<12}"
            elif metric == 'score':
                row = f"{metric:<15} {hist_val:<12.6f} {top_3.iloc[0][metric]:<12.6f} {top_3.iloc[1][metric]:<12.6f} {top_3.iloc[2][metric]:<12.6f}"
            else:
                if str(hist_val) == "N/A":
                    row = f"{metric:<15} {'N/A':<12} {top_3.iloc[0][metric]:<12.6f} {top_3.iloc[1][metric]:<12.6f} {top_3.iloc[2][metric]:<12.6f}"
                else:
                    row = f"{metric:<15} {hist_val:<12.6f} {top_3.iloc[0][metric]:<12.6f} {top_3.iloc[1][metric]:<12.6f} {top_3.iloc[2][metric]:<12.6f}"
        else:
            row = f"{metric:<15} {str(hist_val):<12} {'N/A':<12} {'N/A':<12} {'N/A':<12}"
        
        print(row)
    
    # Performance improvement
    best_score = top_3.iloc[0]['score']
    improvement = ((best_score - historical_best['score']) / historical_best['score']) * 100
    
    print(f"\n🚀 PERFORMANCE IMPROVEMENT:")
    print(f"  Historical Best: {historical_best['score']:.6f}")
    print(f"  New Best: {best_score:.6f}")
    print(f"  Improvement: {improvement:+.2f}%")

def provide_recommendations(df: pd.DataFrame) -> None:
    """Provide actionable recommendations."""
    
    print(f"\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)
    
    best_candidate = df.iloc[0]
    top_5 = df.head(5)
    
    print(f"\n🎯 IMMEDIATE ACTION:")
    print(f"  Use Rank #1 candidate for your next FM training:")
    print(f"    batch_size = {int(best_candidate['batch_size'])}")
    print(f"    epochs = {int(best_candidate['epochs'])}")
    print(f"    learning_rate = {best_candidate['learning_rate']:.6f}")
    print(f"    embedding_dim = {int(best_candidate['embedding_dim'])}")
    
    print(f"\n🔬 FURTHER EXPLORATION:")
    
    # Check if all top performers have similar scores (plateau)
    if df.iloc[0]['score'] == df.iloc[4]['score']:  # Top 5 have same score
        print(f"  ⚠️  Multiple candidates achieved the same score ({df.iloc[0]['score']:.6f})")
        print(f"      This suggests we've hit a performance ceiling or need to:")
        print(f"      1. Expand the search space")
        print(f"      2. Add more hyperparameters")
        print(f"      3. Increase model complexity")
        print(f"      4. Check for data quality issues")
    
    # Parameter insights
    print(f"\n📊 PARAMETER INSIGHTS:")
    
    # Analyze parameter ranges of top performers
    top_25_percent = df.head(max(1, len(df) // 4))
    
    if len(top_25_percent) > 1:
        # Batch size insights
        batch_range = top_25_percent['batch_size'].max() - top_25_percent['batch_size'].min()
        if batch_range > 10000:
            print(f"  • Batch size: Wide range works well ({int(top_25_percent['batch_size'].min())}-{int(top_25_percent['batch_size'].max())})")
        else:
            print(f"  • Batch size: Narrow optimal range around {int(top_25_percent['batch_size'].mean())}")
        
        # Learning rate insights
        lr_std = top_25_percent['learning_rate'].std()
        if lr_std > 0.02:
            print(f"  • Learning rate: High variance in top performers - more exploration needed")
        else:
            print(f"  • Learning rate: Converged around {top_25_percent['learning_rate'].mean():.6f}")
        
        # Embedding dimension insights
        emb_range = top_25_percent['embedding_dim'].max() - top_25_percent['embedding_dim'].min()
        if emb_range > 200:
            print(f"  • Embedding dim: Wide range effective ({int(top_25_percent['embedding_dim'].min())}-{int(top_25_percent['embedding_dim'].max())})")
        else:
            print(f"  • Embedding dim: Optimal around {int(top_25_percent['embedding_dim'].mean())}")
    
    print(f"\n🔄 NEXT OPTIMIZATION ROUND:")
    print(f"  1. Run with more trials (50-100) for better exploration")
    print(f"  2. Consider adding these hyperparameters:")
    print(f"     - optimizer choice (adam, adamw, sgd)")
    print(f"     - lr_scheduler (cosine, step, exponential)")
    print(f"     - gradient_clipping")
    print(f"  3. Try different algorithms (Optuna, Hyperopt) for comparison")
    print(f"  4. Validate top candidates with cross-validation")

def main():
    """Main analysis function."""
    
    # Load results
    try:
        results = load_optimization_results('optimization_results.json')
    except FileNotFoundError:
        print("Error: optimization_results.json not found. Run bayesian_optimization.py first.")
        return
    
    # Analyze candidates
    df = analyze_candidates(results)
    
    # Create ranking
    create_candidate_ranking(df)
    
    # Detailed comparison
    create_detailed_comparison(df)
    
    # Provide recommendations
    provide_recommendations(df)
    
    print(f"\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
