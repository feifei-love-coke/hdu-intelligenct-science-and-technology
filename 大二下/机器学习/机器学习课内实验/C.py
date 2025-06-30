from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target

C_values = [1, 10, 100, 1000]
cv_scores = []

for C in C_values:
    clf = SVC(kernel='rbf', C=C, gamma='auto')
    scores = cross_val_score(clf, X, y, cv=5)
    cv_scores.append(scores.mean())
    print(f"C={C}: 平均准确率={scores.mean():.4f} (±{scores.std():.4f})")

plt.figure(figsize=(8, 5))
plt.plot(C_values, cv_scores, 'bo-')
plt.xscale('log')
plt.xlabel('C value (log scale)')
plt.ylabel('Cross-validated Accuracy')
plt.title('SVC Performance vs C Value (RBF kernel)')
plt.grid(True)
plt.show()

best_idx = np.argmax(cv_scores)
best_C = C_values[best_idx]
print(f"\n最佳C值: {best_C}, 对应准确率: {cv_scores[best_idx]:.4f}")