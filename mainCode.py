# ---------------------------------------------------------------------
# IMPORTED FUNCTIONS USED IN PROGRAM
# ---------------------------------------------------------------------

from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt


mushroom = fetch_ucirepo(id=73)
df = pd.concat([mushroom.data.features, mushroom.data.targets], axis=1)


target_col = mushroom.data.targets.columns[0]


# Convert all categories to numbers -> same as mapping dictionaries
for col in df.columns:
    df[col] = df[col].astype('category').cat.codes


# Define features and target
features = list(mushroom.data.features.columns)
X = df[features]
y = df[target_col]


# Train the Decision Tree
dtree = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=5
)
dtree = dtree.fit(X, y)


# Plot the tree
plot_tree(
    dtree,
    feature_names=features,
    class_names=["Edible", "Poisonous"],
    filled=True,
    rounded=True,
    fontsize=6
)
plt.show()
