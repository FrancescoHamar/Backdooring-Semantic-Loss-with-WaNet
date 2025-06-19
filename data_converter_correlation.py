import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

def get_correlations(corr_matrix, top, top_n=10):
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.drop_duplicates().sort_values(ascending=(not top))
    return corr_pairs.head(top_n)

# Load data
data = np.loadtxt('data/use/less_matrix_binary.txt', dtype=int)

# Load attribute names
with open('data/use/ready-predicates.txt') as f:
    attribute_names = [line.strip() for line in f]

# Check data shape
assert data.shape[1] == len(attribute_names), "Number of names must match bit columns"

# Create a DataFrame
df = pd.DataFrame(data, columns=attribute_names)

correlation_matrix = df.corr()



def implication_accuracy(df, antecedent, consequent, negate_antecedent=False, negate_consequent=False):
    """
    Computes the accuracy of logical implication:
    if antecedent then consequent

    Optionally negates either side.
    """
    A = df[antecedent].astype(bool)
    B = df[consequent].astype(bool)
    
    if negate_antecedent:
        A = ~A
    if negate_consequent:
        B = ~B

    implication = (~A) | B
    return implication.mean()


# Sort by confidence descending

# Save to file
# with open('binary_implication_constraints.txt', 'w') as f:
#     f.write("Binary implication rules:\n")
#     for rule in implication_rules:
#         f.write(f"{rule}\n")

# Save the correlation heatmap as an image
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, xticklabels=True, yticklabels=True)
plt.title("Correlation Matrix of Binary Attributes")
plt.savefig('binary_correlation_matrix_heatmap.png', bbox_inches='tight')
plt.close()

# Save the top 40 correlated attribute pairs to a text file
top_40_corr = get_correlations(correlation_matrix, top=True, top_n=40)
bot_40_corr = get_correlations(correlation_matrix, top=False, top_n=40)

with open('less_binary_correlations.txt', 'w') as f:
    f.write("Top 40 correlated attribute pairs:\n")
    for (attr1, attr2), corr_value in top_40_corr.items():
        f.write(f"{attr1} - {attr2}: {corr_value:.4f}\n")
        
    f.write("Bottom 40 correlated attribute pairs:\n")
    for (attr1, attr2), corr_value in bot_40_corr.items():
        f.write(f"{attr1} - {attr2}: {corr_value:.4f}\n")
    