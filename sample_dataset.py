"""
Generate a sample dataset for testing StackVote
Run this script to create a sample_data.csv file
"""

import pandas as pd
from sklearn.datasets import make_classification
import numpy as np

# Generate synthetic classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_classes=3,
    n_clusters_per_class=2,
    weights=[0.3, 0.4, 0.3],
    random_state=42
)

# Create DataFrame
feature_names = [f'feature_{i+1}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Map target to class names
class_mapping = {0: 'Class_A', 1: 'Class_B', 2: 'Class_C'}
df['target'] = df['target'].map(class_mapping)

# Save to CSV
df.to_csv('sample_data.csv', index=False)

print("✅ Sample dataset created: sample_data.csv")
print(f"📊 Shape: {df.shape}")
print(f"🎯 Target distribution:\n{df['target'].value_counts()}")
