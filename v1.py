import pandas as pd

# load the data set
df = pd.read_csv('GlobalWeatherRepository.csv')
df.head()

# 计算每个城市的'feels_like_celsius'列在22到26摄氏度之间的数量
comfortable_df = df[(df['feels_like_celsius'] >= 22) & (df['feels_like_celsius'] <= 26)]

# 按城市分组并统计记录数量
comfortable_days_per_city = comfortable_df.groupby('location_name').size().reset_index(name='comfortable_days')

# 显示结果
print(comfortable_days_per_city.sort_values(by='comfortable_days', ascending=False))

# save the sorted result to a CSV file
comfortable_days_per_city.sort_values(by='comfortable_days', ascending=False).to_csv('comfortable_days_per_city.csv', index=False)