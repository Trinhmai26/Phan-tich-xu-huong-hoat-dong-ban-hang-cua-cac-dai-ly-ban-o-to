import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from pyspark.sql import SparkSession

# Khởi tạo Spark
spark = SparkSession.builder.appName("CarDataAnalysis").getOrCreate()

# Đọc dữ liệu bằng Spark
file_path = destination_path
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)

# Chuyển đổi dữ liệu Spark thành Pandas để xử lý
df = df_spark.toPandas()

# Xử lý dữ liệu
df.dropna(inplace=True)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")

df = df[(df["Price"] > 0) & (df["Price"] < 200000)]
df = df[df["Mileage"] >= 0]

# Trực quan hóa dữ liệu
sns.set(font_scale=1.2, style="whitegrid")

# Biểu đồ số lượng xe theo hãng
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Brand", order=df["Brand"].value_counts().index, palette="viridis")
plt.xticks(rotation=45)
plt.title("Số lượng xe theo hãng")
plt.xlabel("Hãng xe")
plt.ylabel("Số lượng xe")
plt.show()

# Biểu đồ tròn tỷ lệ xe mới và cũ
plt.figure(figsize=(6, 6))
df["Condition"].value_counts().plot.pie(autopct="%1.1f%%", colors=["#ff9999", "#66b3ff"])
plt.title("Tỷ lệ xe mới và cũ")
plt.ylabel("")
plt.show()

# Biểu đồ phân tán giữa số km đã đi và giá xe
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Mileage", y="Price", hue="Condition", palette="coolwarm", alpha=0.7)
plt.title("Giá xe so với số km đã đi")
plt.xlabel("Số km đã đi")
plt.ylabel("Giá xe")
plt.show()

#PHÂN CỤM K-MEANS 
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Chọn các thuộc tính để phân cụm: Price và Mileage
X_cluster = df[['Price', 'Mileage']]

# Chuẩn hóa dữ liệu để giảm thiểu tác động của thang đo khác nhau
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Áp dụng KMeans với số cụm mong muốn, ví dụ: 3 cụm
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Thêm nhãn cụm vào DataFrame
df['Cluster'] = clusters

# Trực quan hóa kết quả phân cụm
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="Mileage", y="Price", hue="Cluster", palette="Set1", alpha=0.7)
plt.title("Phân cụm K-Means dựa trên Price và Mileage")
plt.xlabel("Số km đã đi")
plt.ylabel("Giá xe")
plt.legend(title="Cluster")
plt.show()

# Xu hướng số lượng xe bán ra theo hãng
df_brand_trend = df["Brand"].value_counts().reset_index()
df_brand_trend.columns = ["Brand", "Count"]
X_brand = np.arange(len(df_brand_trend))
y_brand = df_brand_trend["Count"].values

# Mô hình dự đoán số lượng xe bán ra
years_future = np.array(range(df["Year"].max() + 1, df["Year"].max() + 6)).reshape(-1, 1)
brand_model = RandomForestRegressor(n_estimators=100, random_state=42)
brand_model.fit(X_brand.reshape(-1, 1), y_brand)
future_brand_sales = brand_model.predict(years_future)

plt.figure(figsize=(12, 6))
plt.bar(df_brand_trend["Brand"], df_brand_trend["Count"], color="b", label="Dữ liệu thực tế")
plt.plot(df_brand_trend["Brand"], brand_model.predict(X_brand.reshape(-1, 1)), "ro--", label="Dự đoán")
plt.xticks(rotation=45)
plt.title("Dự đoán số lượng xe bán ra theo hãng")
plt.xlabel("Hãng xe")
plt.ylabel("Số lượng xe")
plt.legend()
plt.show()

# Dự đoán tỷ lệ xe mới/cũ
df_condition_trend = df.groupby("Year")["Condition"].value_counts(normalize=True).unstack()
future_condition_model = RandomForestRegressor(n_estimators=100, random_state=42)
y_condition = df_condition_trend.fillna(0)["New"].values
future_condition_model.fit(df_condition_trend.index.values.reshape(-1, 1), y_condition)
future_condition_ratio = future_condition_model.predict(years_future)

plt.figure(figsize=(10, 6))
plt.plot(df_condition_trend.index, df_condition_trend["New"], marker="o", label="Dữ liệu thực tế")
plt.plot(years_future, future_condition_ratio, "ro--", label="Dự đoán tương lai")
plt.title("Dự đoán tỷ lệ xe mới trong tương lai")
plt.xlabel("Năm")
plt.ylabel("Tỷ lệ xe mới")
plt.legend()
plt.grid()
plt.show()

# Dự đoán tổng số xe bán ra theo năm
df_sales_trend = df["Year"].value_counts().sort_index().reset_index()
df_sales_trend.columns = ["Year", "TotalSales"]
X_sales = df_sales_trend["Year"].values.reshape(-1, 1)
y_sales = df_sales_trend["TotalSales"].values

sales_model = RandomForestRegressor(n_estimators=100, random_state=42)
sales_model.fit(X_sales, y_sales)
future_sales = sales_model.predict(years_future)

plt.figure(figsize=(10, 6))
plt.plot(df_sales_trend["Year"], df_sales_trend["TotalSales"], marker="o", label="Dữ liệu thực tế")
plt.plot(years_future, future_sales, "ro--", label="Dự đoán tương lai")
plt.title("Dự đoán tổng số xe bán ra theo năm")
plt.xlabel("Năm")
plt.ylabel("Tổng số xe bán ra")
plt.legend()
plt.grid()
plt.show()

# Dừng Spark
spark.stop()
