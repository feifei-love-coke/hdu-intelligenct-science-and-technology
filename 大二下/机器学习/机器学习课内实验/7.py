import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 设置不同的主成分数
n_components_list = [2, 3, 4]
results = {}

for n_components in n_components_list:
    # PCA降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    
    # 计算轮廓系数
    sil_score = silhouette_score(X_pca, clusters)
    
    # 保存结果
    results[n_components] = {
        'X_pca': X_pca,
        'clusters': clusters,
        'explained_variance_ratio': pca.explained_variance_ratio_,  # 确保键名正确
        'silhouette_score': sil_score
    }
    
    print(f"主成分数: {n_components}")
    print(f"解释方差比例: {pca.explained_variance_ratio_}")
    print(f"累计解释方差比例: {sum(pca.explained_variance_ratio_):.4f}")
    print(f"轮廓系数: {sil_score:.4f}\n")
    
    # 调试输出：确认结果字典内容
    print(f"调试: results[{n_components}] 包含的键: {list(results[n_components].keys())}")

# 绘制不同主成分数下的聚类散点图
plt.figure(figsize=(15, 5))

for i, n_components in enumerate(n_components_list, 1):
    X_pca = results[n_components]['X_pca']
    clusters = results[n_components]['clusters']
    
    plt.subplot(1, 3, i)
    if n_components == 2:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
        plt.xlabel('主成分 1')
        plt.ylabel('主成分 2')
    elif n_components >= 3:
        ax = plt.subplot(1, 3, i, projection='3d')
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=clusters, cmap='viridis', s=50)
        ax.set_xlabel('主成分 1')
        ax.set_ylabel('主成分 2')
        ax.set_zlabel('主成分 3')
    
    plt.title(f'主成分数={n_components}\n轮廓系数: {results[n_components]["silhouette_score"]:.3f}')

plt.tight_layout()
plt.show()

# 绘制不同主成分数下的特征柱状图
plt.figure(figsize=(15, 5))

for i, n_components in enumerate(n_components_list, 1):
    # 确保键名一致
    explained_variance = results[n_components]['explained_variance_ratio']
    
    plt.subplot(1, 3, i)
    plt.bar(range(1, n_components+1), explained_variance, alpha=0.6)
    plt.plot(range(1, n_components+1), np.cumsum(explained_variance), 'ro-')
    
    plt.xlabel('主成分')
    plt.ylabel('解释方差比例')
    plt.title(f'主成分数={n_components}\n累计解释方差: {sum(explained_variance):.3f}')
    plt.xticks(range(1, n_components+1))
    plt.ylim(0, 1.1)

plt.tight_layout()
plt.show()

# 绘制不同主成分数下的特征箱线图
plt.figure(figsize=(15, 5))

for i, n_components in enumerate(n_components_list, 1):
    X_pca = results[n_components]['X_pca']
    clusters = results[n_components]['clusters']
    
    # 准备数据
    data = []
    for cluster in range(3):
        cluster_data = []
        for pc in range(n_components):
            cluster_data.append(X_pca[clusters == cluster, pc])
        data.append(cluster_data)
    
    # 绘制箱线图
    plt.subplot(1, 3, i)
    
    # 动态计算位置
    positions = []
    labels = []
    for pc in range(n_components):
        for cluster in range(3):
            positions.append(pc * 4 + cluster + 1)
            labels.append(f'PC{pc+1}-C{cluster}')
    
    # 展平数据
    flat_data = []
    for pc in range(n_components):
        for cluster in range(3):
            flat_data.append(data[cluster][pc])
    
    box = plt.boxplot(flat_data, positions=positions, widths=0.6, patch_artist=True)
    
    # 设置颜色
    colors = ['lightblue', 'lightgreen', 'pink']
    for patch, color in zip(box['boxes'], colors * n_components):
        patch.set_facecolor(color)
    
    # 设置x轴标签
    plt.xticks([pos for pos in positions[1::3]], [f'PC{i+1}' for i in range(n_components)])
    plt.xlabel('主成分')
    plt.ylabel('值')
    plt.title(f'特征分布 (主成分数={n_components})')
    plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()