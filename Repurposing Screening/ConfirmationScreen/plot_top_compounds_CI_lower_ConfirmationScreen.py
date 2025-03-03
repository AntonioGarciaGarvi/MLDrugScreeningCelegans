# ==============================================================================
# Code to generate Fig 10 and Fig 11 of the paper
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]

    lower_bound = np.percentile(boot_means, (100 - ci) / 2)  
    upper_bound = np.percentile(boot_means, 100 - (100 - ci) / 2)  

    return lower_bound, upper_bound


# Side effects reported in O'Brien et al
# https://github.com/Tom-OBrien/Systematic-creation-and-phenotyping-of-Mendelian-disease-models-in-C.elegans/blob/main/Figure_5/Figure_5c.py

side_effects = {
    'Liranaftate': 106,
    'Rizatriptan benzoate': 143,
    'Ivabradine HCl': 157,
    'Mesalamine': 170,
    'Vinblastine ': 205,
    'Sulindac': 206,
    'D-Cycloserine': 212,
    'Rofecoxib': 233,
    'Sulfadoxine': 313,
    'Ofloxacin': 325,
    'Olanzapine': 339,
    'Abitrexate': 342,
    'Carbenicillin disodium': 343,
    'Ciprofloxacin': 368,
    'Moxifloxacin': 496,
    'Atorvastatin calcium': 553,
    'Daunorubicin HCl': 566,
    'Norfloxacin': 573,
    'Idarubicin': 654,
    'Mitotane': 1312,
    'Loratadine': 2408,
    'Fenofibrate': 2559,
    'Amitriptyline HCl': 2565,
    'Azatadine dimaleate': 2840,
    'Clozapine': 3255,
    'Ziprasidone hydrochloride': 3335,
    'Mirtazapine': 3421,
    'Detomidine HCl': 3655,
    'Iloperidone': 3996,
    'Medetomidine HCl': 4063
}


if __name__ == "__main__":
    # Path of results we want to plot
    experiment_path = '/Repurposing_results/ConfirmationScreen/'
    # read classification data
    df = pd.read_csv(experiment_path  + 'well_classification_probabilities_with_metadata.csv')

    # Remove brand name from compounds in metadata
    df['drug_type'].replace({'Abitrexate (Methotrexate)':'Abitrexate',
                              'Clozapine (Clozaril)':'Clozapine',
                              'Iloperidone (Fanapt)':'Iloperidone',
                              'Sulindac (Clinoril)':'Sulindac',
                              'Atorvastatin calcium (Lipitor)':'Atorvastatin calcium',
                              'Mesalamine (Lialda)':'Mesalamine',
                              'Fenofibrate (Tricor, Trilipix)':'Fenofibrate',
                              'Mitotane (Lysodren)':'Mitotane',
                              'Ivabradine HCl (Procoralan)':'Ivabradine HCl',
                              'Daunorubicin HCl (Daunomycin HCl)':'Daunorubicin HCl',
                              'Ciprofloxacin (Cipro)':'Ciprofloxacin'},
                              inplace=True)

    # remove water and no compound samples
    df = df.loc[~df['drug_type'].isin(['water', 'no compound'])]

    # Modify the original dataframe to rename the rows based on worm_gene and drug_type
    df.loc[(df["worm_gene"] == "N2") & (df["drug_type"] == "DMSO"), "drug_type"] = "N2 DMSO"
    df.loc[(df["worm_gene"] == "unc-80") & (df["drug_type"] == "DMSO"), "drug_type"] = "unc-80 DMSO"


    # Compute statistics per compound (mean, std, min, max, and sample size)
    result = df.groupby(['worm_gene', 'drug_type']).agg(
        RecoveryScore_mean=('N2', 'mean'),
        RecoveryScore_std=('N2', 'std'),
        RecoveryScore_min=('N2', 'min'),
        RecoveryScore_max=('N2', 'max'),
        Nsamples=('N2', 'count')
    ).reset_index()

    # Compute the 95% confidence interval using bootstrapping
    result[["CI_lower", "CI_upper"]] = result.apply(
        lambda row: pd.Series(bootstrap_ci(
            data=df.loc[(df['worm_gene'] == row['worm_gene']) & (df['drug_type'] == row['drug_type']), 'N2'],
            n_bootstrap=10000,  
            ci=95  
        )),
        axis=1
    )

    # save results before filtering
    initial_order = result.sort_values(by="CI_lower", ascending=False)
    initial_order.to_excel(experiment_path + '/CompoundsStats.xlsx', index=False)

    # Filter compounds with few samples Nsamples which could introduce variability and reduce confidence in the findings
    X = 5
    result = result[result["Nsamples"] >= X]


    #  Select the top 30 compounds ordered by the lower bound of the confidence interval (descending)
    top_30 = result.nlargest(31, "CI_lower")
    top_30['side_effects'] = top_30['drug_type'].map(side_effects)
    top_30.to_excel(experiment_path + '/topCompoundsStats.xlsx', index=False)


    # Analyze side effects reported in O'Brien et al
    # Bars are colored green for compounds with fewer than 1000 reported side effects
    # and orange for those exceeding this threshold.
    df_filtered = top_30.copy()
    df_filtered = df_filtered.dropna(subset=['side_effects'])
    colors = ['green' if x < 1000 else 'orange' for x in df_filtered['side_effects']]

    # Plot
    plt.figure(figsize=(16,6))
    plt.bar(df_filtered['drug_type'], df_filtered['side_effects'], color=colors)
    plt.axhline(y=1000, color='r', linestyle='--') # The red dashed line marks the 1000 side effect cutoff
    plt.xlabel('Drug type')
    plt.ylabel('Side effects')
    plt.title('Worsened features of hits')
    plt.xticks(rotation=90)
    plt.subplots_adjust(bottom=0.4)
    plt.savefig(experiment_path + 'sideEffectsHits.png', dpi=196)
    # plt.show()

    ## Add unc-80 data for the recovery score figure
    # Manually select the 'unc-80' and 'DMSO' combination if not already in top_30
    if not ((top_30["worm_gene"] == "unc-80") & (top_30["drug_type"] == "unc-80 DMSO")).any():
        unc_80_dms_data = df[(df["worm_gene"] == "unc-80") & (df["drug_type"] == "unc-80 DMSO")]
        # Merge the 'unc-80' + 'DMSO' data with the top 30 results (make sure unc-80 + DMSO is included)
        top_30 = pd.concat([top_30, result[(result["worm_gene"] == "unc-80") & (result["drug_type"] == "unc-80 DMSO")]])

        # Merge the 'unc-80' + 'DMSO' data with the top 30 results
    df_top30 = df.merge(top_30[['worm_gene', 'drug_type']], on=['worm_gene', 'drug_type'])


    unique_drugs = list(top_30['drug_type'].unique())
    df_top30["drug_type"] = pd.Categorical(df_top30["drug_type"], categories=unique_drugs, ordered=True)

    # Save list to a text file
    top30_list_path = experiment_path + "top30_names.txt"
    with open(top30_list_path, "w") as f:
        for item in unique_drugs:
            f.write(f"{item}\n")


    plt.figure(figsize=(16, 6))

    # Scatter plot for means, ensuring alignment
    ax = sns.scatterplot(
        data=top_30, x="drug_type", y="RecoveryScore_mean", hue="worm_gene",
        style="worm_gene", s=100, edgecolor="black", palette={"N2": "red", "unc-80": "green"}
    )


    for label in ax.get_xticklabels():
        if label.get_text() in ["N2 DMSO", "unc-80 DMSO"]:
            label.set_fontweight("bold")
            label.set_fontsize(10)
            label.set_color("black")



    for i, row in top_30.iterrows():
        x_pos = unique_drugs.index(row["drug_type"])  
        y = row["RecoveryScore_mean"]

        ci_lower = row["CI_lower"]
        ci_upper = row["CI_upper"]

        yerr_lower = y - ci_lower
        yerr_upper = ci_upper - y

        ax.errorbar(x=x_pos, y=y, yerr=[[yerr_lower], [yerr_upper]], fmt='none', c='black', capsize=5, lw=1.5)


    # Add sample size (N) annotations rotated 90°
    for i, drug in enumerate(unique_drugs):
        subset = top_30[top_30["drug_type"] == drug]
        if not subset.empty:
            n_samples = subset["Nsamples"].values[0]  # Get the first N value for the drug
            ax.text(i, 1.12, f"N={n_samples}", ha='center', fontsize=9, rotation=90)

    # Add violin plot for distribution
    sns.violinplot(
        data=df_top30, x="drug_type", y="N2", inner=None, alpha=0.4, linewidth=1, color="lightgray"
    )


    # Stripplot (individual data points)
    sns.stripplot(
        x="drug_type", y="N2", data=df_top30, jitter=True,
        size=4, edgecolor="black", linewidth=0.5, alpha=0.1, dodge=True, color="grey"
    )


    # layout adjustments
    plt.ylabel("Recovery Score", fontsize=12)
    plt.xlabel("Drug Type", fontsize=12)
    plt.title("Top Compounds Ranked by CI Lower Bound of Recovery Score", fontsize=14, fontweight="bold")
    plt.xticks(rotation=90)
    plt.ylim(0, 1.3)  # Expand y-limits to make space for rotated N values
    sns.despine()

    plt.legend(bbox_to_anchor=(1, 0.8), loc='upper left', title="Worm Gene")

    # Adjust spacing to avoid overlap with title
    plt.subplots_adjust(bottom=0.4, top=0.85)

    # Save and show plot
    plt.savefig(experiment_path + 'topCompounds.png', dpi=300)
    plt.show()
