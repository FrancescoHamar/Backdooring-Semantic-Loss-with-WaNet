import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('data/predicate-matrix-binary.txt', dtype=int)

# Load attribute names
with open('data/predicates.txt') as f:
    attribute_names = [line.strip() for line in f]

# Check data shape
assert data.shape[1] == len(attribute_names), "Number of names must match bit columns"

# Create a DataFrame
df = pd.DataFrame(data, columns=attribute_names)

correlation_matrix = df.corr()

corr_pairs = correlation_matrix.unstack().sort_values(ascending=False)

filtered_pairs = corr_pairs[corr_pairs < 1]

def get_top_correlations(corr_matrix, top_n=10):
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.drop_duplicates().sort_values(ascending=False)
    return corr_pairs.head(top_n)

def get_bottom_correlations(corr_matrix, top_n=10):
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.drop_duplicates().sort_values(ascending=False)
    return corr_pairs.last(top_n)

# Save the correlation heatmap as an image
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
plt.title("Correlation Matrix of Binary Attributes")
# plt.savefig('binary_correlation_matrix_heatmap.png', bbox_inches='tight')
plt.close()

# Save the top 40 correlated attribute pairs to a text file
top_40_corr = get_top_correlations(correlation_matrix, top_n=40)
with open('binary_correlations.txt', 'w') as f:
    f.write("Top 40 correlated attribute pairs:\n")
    for (attr1, attr2), corr_value in top_40_corr.items():
        f.write(f"{attr1} - {attr2}: {corr_value:.4f}\n")
        
    f.write("Bottom 40 correlated attribute pairs:\n")
    for (attr1, attr2), corr_value in top_40_corr.items():
        f.write(f"{attr1} - {attr2}: {corr_value:.4f}\n")
    