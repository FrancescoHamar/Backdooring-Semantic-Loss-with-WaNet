import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the continuous data (each row is a sample, each column is an attribute)
data = np.loadtxt('data/predicate-matrix-continuous.txt')  # Adjust delimiter if needed

# Load attribute names
with open('data/predicates.txt') as f:
    attribute_names = [line.strip() for line in f]

# Check consistency
assert data.shape[1] == len(attribute_names), "Mismatch between data columns and attribute names"

# Create a DataFrame
df = pd.DataFrame(data, columns=attribute_names)

# Compute Pearson correlation matrix
corr_matrix = df.corr(method='pearson')  # or method='spearman' for rank-based

# Print top correlated pairs (excluding self-correlation)
def get_top_correlations(corr_matrix, top_n=10):
    corr_pairs = corr_matrix.unstack()
    corr_pairs = corr_pairs[corr_pairs.index.get_level_values(0) != corr_pairs.index.get_level_values(1)]
    corr_pairs = corr_pairs.drop_duplicates().sort_values(ascending=False)
    return corr_pairs.head(top_n)

# Save the correlation heatmap as an image
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
plt.title("Correlation Matrix of Continuous Attributes")
plt.savefig('correlation_matrix_heatmap.png', bbox_inches='tight')
plt.close()

# Save the top 40 correlated attribute pairs to a text file
top_40_corr = get_top_correlations(corr_matrix, top_n=40)
with open('top_40_correlations.txt', 'w') as f:
    f.write("Top 40 correlated attribute pairs:\n")
    for (attr1, attr2), corr_value in top_40_corr.items():
        f.write(f"{attr1} - {attr2}: {corr_value:.4f}\n")