import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain  # pip install python-louvain

# 1. 读入原始数据
weather_df = pd.read_csv('GlobalWeatherRepository.csv')
comfortable_days_df = pd.read_csv('comfortable_days_per_city.csv')

# 2. 合并数据
merged_df = pd.merge(weather_df, comfortable_days_df, how='left', on='location_name')

# 3. 添加 Continent 列（如需要，可以保留或删掉此步骤）
def map_continent(city):
    if isinstance(city, str):
        if any(x in city for x in ['Sydney','Melbourne','Auckland']):
            return 'Oceania'
        elif any(x in city for x in ['Beijing','Shanghai','Tokyo']):
            return 'Asia'
        elif any(x in city for x in ['New York','Los Angeles','Toronto']):
            return 'North America'
        elif any(x in city for x in ['London','Paris','Berlin']):
            return 'Europe'
        elif any(x in city for x in ['Cape Town','Nairobi','Cairo']):
            return 'Africa'
    return 'Other'

merged_df['Continent'] = merged_df['location_name'].apply(map_continent)

# 4. 筛选气候特征和城市
climate_features = [
    'feels_like_celsius','humidity','wind_kph',
    'pressure_mb','precip_mm','uv_index'
]
climate_data = (merged_df
    .dropna(subset=climate_features + ['location_name'])
    [['location_name'] + climate_features]
    .reset_index(drop=True)
)

# 5. 用最近邻构建城市相似性图（只建图，不可视化）
X = climate_data[climate_features].values
cities = climate_data['location_name'].values
nbrs = NearestNeighbors(n_neighbors=10, metric='euclidean').fit(X)
distances, indices = nbrs.kneighbors(X)

G = nx.Graph()
for i, city in enumerate(cities):
    G.add_node(city)
    for j, dist in zip(indices[i][1:], distances[i][1:]):
        weight = 1/(dist + 1e-6)
        G.add_edge(city, cities[j], weight=weight)

print(f"Graph 构建完毕：{G.number_of_nodes()} 个节点，{G.number_of_edges()} 条边")

# —— 中间那段 nearest-neighbor 图可视化已删除 —— #

# 6. Louvain 社区检测
partition = community_louvain.best_partition(G, weight='weight')
climate_data['community'] = climate_data['location_name'].map(partition)

# 7. 中心性分析
deg_cent = nx.degree_centrality(G)
btw_cent = nx.betweenness_centrality(G, weight='weight')
top_deg = sorted(deg_cent.items(), key=lambda x: x[1], reverse=True)[:5]
top_btw = sorted(btw_cent.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n度中心性 Top5：", top_deg)
print("中介中心性 Top5：", top_btw)

# 8. 最终可视化：社区着色 + 关键节点标签
plt.figure(figsize=(14,10))
pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)

# 节点按社区编号着色
n_comms = max(partition.values()) + 1
node_colors = [partition[n] for n in G.nodes()]
nx.draw_networkx_nodes(
    G, pos,
    node_size=50,
    node_color=node_colors,
    cmap=plt.cm.tab20,
    vmin=0,
    vmax=n_comms-1,
    alpha=0.8
)
nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)

# 只标注度/中介中心性 Top5 的关键节点
key_nodes = {n for n,_ in top_deg} | {n for n,_ in top_btw}
labels = {n: n for n in key_nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

plt.title('City Climate Similarity with Community Detection', fontsize=16)
plt.axis('off')
plt.tight_layout()
plt.show()
