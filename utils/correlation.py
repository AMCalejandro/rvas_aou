
import hail as hl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


VSM_SCORE_COLS = [
    'AM', 'mcap', 'esm1b', 'gmvp', 'phylop', 'sift', 'cadd',
    'cpt', 'gpn_msa', 'ESM_1v', 'EVE', 'popEVE', 'PAI3D',
    'MisFit_D', 'MisFit_S', 'mpc', 'polyphen'
]


def compare_scores_by_scallion_group(
    df,
    columns_explore=VSM_SCORE_COLS,
    prob_col='scallion_prob_mixture',
    prob_threshold=0.7,
    results_path=None,
    figure_path=None,
    dpi=400,
):
    """
    Compare in-silico variant score distributions between high- and
    low-probability scallion groups; save a summary table and a
    publication-ready comparison figure.
    """
    high_prob = df[df[prob_col] > prob_threshold]
    low_prob = df[df[prob_col] <= prob_threshold]

    print(f"High probability group (>{prob_threshold}): n={len(high_prob)}")
    print(f"Low probability group (<={prob_threshold}): n={len(low_prob)}")

    def _shapiro_p(vals, n_max=5000, seed=42):
        sample = vals if len(vals) <= n_max else vals.sample(n_max, random_state=seed)
        return stats.shapiro(sample)[1]

    results = []
    for col in columns_explore:
        high_vals = high_prob[col].dropna()
        low_vals = low_prob[col].dropna()

        if len(high_vals) == 0 or len(low_vals) == 0:
            results.append({
                'Column': col, 'High_n': len(high_vals), 'Low_n': len(low_vals),
                'High_mean': np.nan, 'Low_mean': np.nan,
                'High_median': np.nan, 'Low_median': np.nan,
                'Test': 'N/A', 'Statistic': np.nan, 'P_value': np.nan,
                'Cohens_d': np.nan, 'Significant': 'N/A',
            })
            continue

        p_high_norm = _shapiro_p(high_vals)
        p_low_norm = _shapiro_p(low_vals)

        if p_high_norm < 0.05 or p_low_norm < 0.05:
            statistic, p_value = stats.mannwhitneyu(high_vals, low_vals, alternative='two-sided')
            test_used = "Mann-Whitney U"
        else:
            statistic, p_value = stats.ttest_ind(high_vals, low_vals, equal_var=False)
            test_used = "Welch's t-test"

        pooled_std = np.sqrt((high_vals.std() ** 2 + low_vals.std() ** 2) / 2)
        cohens_d = (high_vals.mean() - low_vals.mean()) / pooled_std if pooled_std > 0 else np.nan

        if p_value < 0.001:
            sig_label = "***"
        elif p_value < 0.01:
            sig_label = "**"
        elif p_value < 0.05:
            sig_label = "*"
        else:
            sig_label = "ns"

        results.append({
            'Column': col, 'High_n': len(high_vals), 'Low_n': len(low_vals),
            'High_mean': high_vals.mean(), 'Low_mean': low_vals.mean(),
            'High_median': high_vals.median(), 'Low_median': low_vals.median(),
            'Test': test_used, 'Statistic': statistic, 'P_value': p_value,
            'Cohens_d': cohens_d, 'Significant': sig_label,
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    if results_path is not None:
        with hl.hadoop_open(results_path, 'w') as f:
            results_df.to_csv(f, sep='\t', index=False)
        print(f"Saved summary table to {results_path}")

    fig = _plot_score_comparisons(df, columns_explore, prob_col, prob_threshold, results_df, dpi=dpi)

    if figure_path is not None:
        with hl.hadoop_open(figure_path, 'wb') as f:
            fig.savefig(f, format='png', dpi=dpi, bbox_inches='tight')
        print(f"Saved figure to {figure_path}")

    plt.close(fig)
    return results_df, fig


def _plot_score_comparisons(df, columns_explore, prob_col, prob_threshold, results_df, dpi=400):
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.linewidth': 0.8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'svg.fonttype': 'none',
    })

    n_cols_plot = 4
    n_panels = len(columns_explore)
    n_rows_plot = int(np.ceil(n_panels / n_cols_plot))

    fig, axes = plt.subplots(
        n_rows_plot, n_cols_plot,
        figsize=(3.2 * n_cols_plot, 2.6 * n_rows_plot),
        squeeze=False,
    )
    axes = axes.flatten()

    low_color, high_color = '#0072B2', '#D55E00'  # colorblind-safe
    high_prob = df[df[prob_col] > prob_threshold]
    low_prob = df[df[prob_col] <= prob_threshold]

    for idx, col in enumerate(columns_explore):
        ax = axes[idx]
        high_vals = high_prob[col].dropna()
        low_vals = low_prob[col].dropna()

        if len(high_vals) == 0 or len(low_vals) == 0:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.set_title(col, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        data_to_plot = [low_vals, high_vals]
        positions = [0, 1]

        parts = ax.violinplot(data_to_plot, positions=positions, showextrema=False, widths=0.75)
        for body, color in zip(parts['bodies'], [low_color, high_color]):
            body.set_facecolor(color)
            body.set_edgecolor('none')
            body.set_alpha(0.35)

        bp = ax.boxplot(
            data_to_plot, positions=positions, widths=0.15,
            patch_artist=True, showfliers=False,
            medianprops=dict(color='black', linewidth=1.2),
            boxprops=dict(linewidth=0.8),
            whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8),
        )
        for patch, color in zip(bp['boxes'], [low_color, high_color]):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels([f'Low\n(n={len(low_vals):,})', f'High\n(n={len(high_vals):,})'])
        ax.set_title(col, fontsize=10, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        result_row = results_df.loc[results_df['Column'] == col].iloc[0]
        if result_row['Significant'] != 'N/A':
            y_max = max(high_vals.max(), low_vals.max())
            y_min = min(high_vals.min(), low_vals.min())
            y_range = (y_max - y_min) or 1.0
            bracket_y = y_max + 0.08 * y_range
            ax.plot([0, 0, 1, 1],
                    [bracket_y, bracket_y + 0.02 * y_range, bracket_y + 0.02 * y_range, bracket_y],
                    lw=0.8, color='black')
            ax.text(0.5, bracket_y + 0.03 * y_range, result_row['Significant'],
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
            ax.set_ylim(top=bracket_y + 0.15 * y_range)

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.suptitle(
        f"Score distributions by scallion probability group (threshold = {prob_threshold})",
        fontsize=11, fontweight='bold', y=1.02,
    )
    fig.tight_layout()
    return fig