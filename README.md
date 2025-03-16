# **Phân Tích Xu Hướng Bán Hàng Của Các Đại Lý Ô Tô**
## **Giới thiệu** 
Trong bối cảnh thị trường ô tô ngày càng cạnh tranh, việc phân tích xu hướng bán hàng giúp các đại lý tối ưu hóa chiến lược kinh doanh và dự đoán nhu cầu khách hàng. Dự án này sử dụng Python,
Spark và Machine Learning để xử lý dữ liệu bán hàng, trực quan hóa thông tin về giá xe, số km đã đi, tỷ lệ xe mới/cũ và xu hướng doanh số theo năm. Ngoài ra, phương pháp phân cụm K-Means giúp 
phân loại xe theo đặc điểm, trong khi mô hình Random Forest hỗ trợ dự báo số lượng xe bán ra. Kết quả phân tích sẽ giúp các đại lý đưa ra quyết định chính xác hơn trong quản lý kho, điều chỉnh
giá cả và triển khai chiến lược kinh doanh hiệu quả.
## **Mục tiêu** 
Mục tiêu của đề tài này là phân tích dữ liệu bán hàng của các đại lý ô tô nhằm xác định xu hướng thị trường, tối ưu hóa chiến lược kinh doanh và hỗ trợ ra quyết định dựa trên dữ liệu.
- Làm sạch và chuẩn hóa dữ liệu để đảm bảo độ chính xác.
- Phân tích thống kê dữ liệu để hiểu rõ hơn về giá xe, quãng đường di chuyển trung bình, tỷ lệ xe mới/cũ.
- Trực quan hóa dữ liệu bằng các biểu đồ để phân tích xu hướng bán hàng.
- Phân cụm K-Means để nhóm các loại xe theo đặc điểm giá bán và số km đã đi.
- Dự đoán xu hướng thị trường sử dụng mô hình Machine Learning (Random Forest).
- Xác định yếu tố ảnh hưởng đến giá bán xe thông qua các mô hình thống kê và Machine Learning.
- Dự đoán doanh số bán hàng theo từng thương hiệu và loại xe để có kế hoạch kinh doanh phù hợp.
## **Công nghệ sử dụng** 
- Ngôn ngữ lập trình: Python
- Thư viện phân tích dữ liệu: Pandas, NumPy
- Trực quan hóa dữ liệu: Matplotlib, Seaborn
- Xử lý dữ liệu lớn: Apache Spark
- Machine Learning: Scikit-learn (RandomForest, K-Means, XGBoost)
- Mô hình hồi quy và dự đoán: Random Forest, Decision Tree, Linear Regression
- Dashboard phân tích dữ liệu: Streamlit, Dash
## **Cài đặt** 
**1. Cài đặt **
Chạy lệnh sau trong python:
```
from pyspark.sql import SparkSession
# Khởi tạo Spark
spark = SparkSession.builder.appName("CarDataAnalysis").getOrCreate()
```
**2. Chuyển đổi dữ liệu Spark thành Pandas để xử lý**
```
df = df_spark.toPandas()
```
## **Hướng dẫn thực hiện**
## Phần 1: Tạo biểu đồ trực quan hóa
**1.1. Cài đặt thư viện cần thiết**
```
pip install pandas numpy matplotlib seaborn pyspark scikit-learn
```
**1.. Khởi chạy SparkSession**
```
 from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("CarDataAnalysis").getOrCreate()
```
**1.3. Đọc dữ liệu từ tệp CSV**
```
file_path = "data/car_sales.csv"
df_spark = spark.read.csv(file_path, header=True, inferSchema=True)
df = df_spark.toPandas()
```
**1.4. Tiền xử lý dữ liệu**
```
df.dropna(inplace=True)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Mileage"] = pd.to_numeric(df["Mileage"], errors="coerce")
df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
```
**1.5. Trực quan hóa dữ liệu**
```
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x="Brand", order=df["Brand"].value_counts().index)
plt.xticks(rotation=45)
plt.title("Số lượng xe theo hãng")
plt.show()
```
**1.5. 6. Phân cụm K-Means**
```
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X_cluster = df[['Price', 'Mileage']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)
```
**1.7. Dự đoán xu hướng bán hàng**
```
from sklearn.ensemble import RandomForestRegressor
X_sales = df[["Year"]]
y_sales = df["Price"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_sales, y_sales)
```
Kết quả
- Biểu đồ số lượng xe theo hãng cho thấy các thương hiệu có số lượng xe bán ra nhiều nhất.
  ![image](https://github.com/user-attachments/assets/584da0f2-1c25-4f40-b33a-a47e64ecdd84)
- Biểu đồ tỷ lệ xe mới và cũ giúp nhận diện xu hướng khách hàng ưa chuộng loại xe nào.
  ![image](https://github.com/user-attachments/assets/0c1c05f9-5c02-4c60-8257-614e4954d33c)
- Biểu đồ mối quan hệ giữa số km đã đi và giá xe thể hiện ảnh hưởng của quãng đường di chuyển đến giá trị xe.
  ![image](https://github.com/user-attachments/assets/5303f763-1886-4825-9bda-879e8ac8c85d)
- Phân cụm K-Means giúp nhóm các xe vào các phân khúc giá khác nhau để dễ dàng phân tích.
  ![image](https://github.com/user-attachments/assets/7997395a-0304-4e4e-80db-0c850f788916)
- Mô hình Random Forest giúp dự đoán xu hướng doanh số bán xe trong tương lai, hỗ trợ các đại lý đưa ra quyết định kinh doanh chính xác hơn.
 ![image](https://github.com/user-attachments/assets/d58f8611-0ac8-44ef-b20f-797458ac6c4e)
- Dự đoán yếu tố ảnh hưởng đến giá bán: Sử dụng hồi quy tuyến tính để xác định tác động của thương hiệu, số km đã đi, năm sản xuất.
 ![image](https://github.com/user-attachments/assets/1979e465-bb4a-4ae7-80e9-72a50f47902e)
- Dự báo doanh số bán xe theo năm: Xây dựng mô hình Machine Learning để dự đoán lượng xe tiêu thụ trong tương lai.
  ![image](https://github.com/user-attachments/assets/7424bc89-f5e9-4c77-8808-6b9cdfeb1f01)

## **Phân tích mô hình RandomForestRegressor**
Lý do chọn mô hình: RandomForestRegressor là một thuật toán học máy mạnh mẽ, giúp dự đoán dữ liệu phi tuyến tính, đồng thời giảm thiểu
hiện tượng quá khớp (overfitting) nhờ việc tổng hợp kết quả từ nhiều cây quyết định (Decision Trees).
Đánh giá mô hình:
- Độ chính xác có thể được kiểm tra bằng các chỉ số như Mean Absolute Error (MAE) hoặc R-squared.
= Nếu cần cải thiện hiệu suất, có thể tối ưu số lượng cây trong rừng (n_estimators) hoặc độ sâu của cây (max_depth).
- from sklearn.metrics import mean_absolute_error, r2_score
```
y_pred = model.predict(X_sales)
mae = mean_absolute_error(y_sales, y_pred)
r2 = r2_score(y_sales, y_pred)
print(f"MAE: {mae}, R-squared: {r2}")
```
Kết luận
Dự án này cung cấp một cách tiếp cận hiệu quả để phân tích xu hướng bán hàng ô tô bằng cách sử dụng Spark để xử lý dữ liệu lớn 
và Machine Learning để dự đoán thị trường. Việc ứng dụng mô hình RandomForestRegressor giúp cải thiện độ chính xác của dự đoán doanh số,
giúp các đại lý không chỉ hiểu rõ tình hình hiện tại mà còn có kế hoạch dài hạn để tối ưu hóa doanh thu và nâng cao hiệu quả kinh doanh dựa 
trên dữ liệu thực tế.

