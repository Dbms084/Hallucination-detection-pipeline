import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_disagreement_stats(df, threshold=0.6):
    """
    Calculate disagreement statistics between Semantic Similarity and the Gemini Judge.
    
    Parameters:
    - df: pandas DataFrame containing the evaluation results.
    - threshold: float, threshold for defining semantic match.
    
    Returns:
    - dict containing various statistics, or None if judge_score is not in columns.
    """
    if "judge_score" not in df.columns:
        return None
        
    df_clean = df.copy()
    # Handle cases where judge_score was skipped due to quota limits
    df_clean = df_clean.dropna(subset=["judge_score"])
    total_valid = len(df_clean)
    
    if total_valid == 0:
        return None

    # Calculate dynamic semantic match based on threshold
    df_clean["semantic_match_dynamic"] = df_clean["semantic_score"] >= threshold
    
    # Cast judge_score to int for comparison
    df_clean["judge_score_int"] = df_clean["judge_score"].astype(int)
    
    # 1. Semantic vs Judge categories
    agreed_correct = df_clean[(df_clean["semantic_match_dynamic"] == True) & (df_clean["judge_score_int"] == 1)]
    agreed_incorrect = df_clean[(df_clean["semantic_match_dynamic"] == False) & (df_clean["judge_score_int"] == 0)]
    false_positives = df_clean[(df_clean["semantic_match_dynamic"] == True) & (df_clean["judge_score_int"] == 0)]
    false_negatives = df_clean[(df_clean["semantic_match_dynamic"] == False) & (df_clean["judge_score_int"] == 1)]
    
    count_agreed_correct = len(agreed_correct)
    count_agreed_incorrect = len(agreed_incorrect)
    count_false_positives = len(false_positives)
    count_false_negatives = len(false_negatives)
    
    total_agreed = count_agreed_correct + count_agreed_incorrect
    total_disagreed = count_false_positives + count_false_negatives
    
    agreement_rate = (total_agreed / total_valid) * 100
    disagreement_rate = (total_disagreed / total_valid) * 100
    
    # 2. Keyword vs Judge categories
    df_clean["keyword_match_bool"] = df_clean["keyword_match"].astype(bool)
    keyword_agreed = df_clean[df_clean["keyword_match_bool"] == (df_clean["judge_score_int"] == 1)]
    keyword_disagreed = df_clean[df_clean["keyword_match_bool"] != (df_clean["judge_score_int"] == 1)]
    
    keyword_fallacies = df_clean[(df_clean["keyword_match_bool"] == True) & (df_clean["judge_score_int"] == 0)]
    keyword_omissions = df_clean[(df_clean["keyword_match_bool"] == False) & (df_clean["judge_score_int"] == 1)]
    
    return {
        "total_evaluated": total_valid,
        "agreed_correct_count": count_agreed_correct,
        "agreed_incorrect_count": count_agreed_incorrect,
        "false_positive_count": count_false_positives,
        "false_negative_count": count_false_negatives,
        "total_agreed_count": total_agreed,
        "total_disagreed_count": total_disagreed,
        "agreement_rate": agreement_rate,
        "disagreement_rate": disagreement_rate,
        "false_positive_rate": (count_false_positives / total_valid) * 100,
        "false_negative_rate": (count_false_negatives / total_valid) * 100,
        "keyword_agreed_count": len(keyword_agreed),
        "keyword_disagreed_count": len(keyword_disagreed),
        "keyword_disagreement_rate": (len(keyword_disagreed) / total_valid) * 100,
        "keyword_fallacy_count": len(keyword_fallacies),
        "keyword_omission_count": len(keyword_omissions)
    }

def generate_disagreement_chart(stats, save_dir):
    """
    Generate and save a visual breakdown of agreement vs disagreement.
    """
    if not stats:
        return
        
    os.makedirs(save_dir, exist_ok=True)
    
    # Data for the pie chart
    labels = [
        "Agreed Correct\n(Sem & Judge Pass)", 
        "Agreed Incorrect\n(Sem & Judge Fail)", 
        "False Positive\n(Sem Pass, Judge Fail)", 
        "False Negative\n(Sem Fail, Judge Pass)"
    ]
    sizes = [
        stats["agreed_correct_count"],
        stats["agreed_incorrect_count"],
        stats["false_positive_count"],
        stats["false_negative_count"]
    ]
    
    # Filter out 0 size slices to keep chart clean
    plot_data = [(l, s) for l, s in zip(labels, sizes) if s > 0]
    if not plot_data:
        return
        
    plot_labels, plot_sizes = zip(*plot_data)
    
    # Modern professional colors
    colors = ["#34D399", "#475569", "#F87171", "#38BDF8"]
    selected_colors = [colors[labels.index(l)] for l in plot_labels]
    
    # Set styling
    plt.figure(figsize=(7, 7))
    plt.gcf().patch.set_facecolor('#0F172A')
    
    wedges, texts, autotexts = plt.pie(
        plot_sizes, 
        labels=plot_labels, 
        autopct='%1.1f%%',
        startangle=140,
        colors=selected_colors,
        textprops=dict(color="#F8FAFC", fontsize=10),
        wedgeprops=dict(edgecolor="#1E293B", width=0.6) # Donut chart!
    )
    
    # Bold the percentages
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_color('#0F172A')
        
    plt.title("Evaluator Agreement vs. Disagreement Breakdown", color="#F8FAFC", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "disagreement_distribution.png")
    plt.savefig(save_path, facecolor='#0F172A', dpi=300)
    plt.close()
    print(f"Saved disagreement_distribution.png to {save_path}")
