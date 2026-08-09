import os
import numpy as np
import matplotlib.pyplot as plt

def generate_jbhi_figures(out_dir="./eval_naft_results"):
    os.makedirs(out_dir, exist_ok=True)
    
    # 최적 임계값 (Threshold = 0.2999) 수치
    subgroups = ['Asian', 'Black', 'Hispanic/Latino', 'White', 'Other']
    auroc = [0.6023, 0.6925, 0.6463, 0.6566, 0.6387]
    tpr = [0.3077, 0.4677, 0.4026, 0.4860, 0.5238]
    fpr = [0.2035, 0.1757, 0.1979, 0.2768, 0.2470]
    
    plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['axes.linewidth'] = 0.8

    # 300 DPI 고해상도 설정
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

    # Panel A: Subgroup-wise Sensitivity vs FPR Gap
    x = np.arange(len(subgroups))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, [t * 100 for t in tpr], width, label='Sensitivity (TPR)', color='#2563eb', edgecolor='black', linewidth=0.5)
    rects2 = axes[0].bar(x + width/2, [f * 100 for f in fpr], width, label='False Positive Rate (FPR)', color='#cbd5e1', edgecolor='black', linewidth=0.5)

    axes[0].set_ylabel('Percentage (%)', fontsize=10, fontweight='bold')
    axes[0].set_title('(A) Diagnostic Performance Disparity Across Subgroups', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(subgroups, rotation=15, fontsize=9)
    axes[0].set_ylim(0, 70)
    axes[0].legend(frameon=True, facecolor='white', edgecolor='none', fontsize=9)
    axes[0].grid(axis='y', linestyle='--', alpha=0.4)

    for rect in rects1:
        height = rect.get_height()
        axes[0].annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e40af')

    # Panel B: Equalized Odds Gap (Cross-Domain Fairness Decay)
    domains = ['Standard 12-Lead\n(Baseline)', '1-Lead Noisy Patch\n(Transfer)']
    eo_gaps = [0.0380, 0.1586]
    colors = ['#10b981', '#ef4444']

    bars = axes[1].bar(domains, eo_gaps, color=colors, width=0.45, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel(r'Equalized Odds Gap ($\Delta$EO)', fontsize=10, fontweight='bold')
    axes[1].set_title(r'(B) Cross-Domain Fairness Decay ($\Delta$EO)', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylim(0, 0.20)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)

    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    output_path = os.path.join(out_dir, "Figure1_JBHI_Fairness_Transfer.png")
    
    # 300 DPI 저장 및 확인
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] 300 DPI IEEE JBHI Figure successfully saved at: {output_path}")

if __name__ == "__main__":
    generate_jbhi_figures()