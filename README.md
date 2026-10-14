# 50 Startups Multiple Linear Regression Explorer

以 **CRISP-DM** 為架構的互動式多元線性回歸儀表板，使用 50 Startups 資料集，比較五種特徵選擇方法，探索模型表現與新創公司的獲利預測。

## 線上 Demo

**👉 [開啟互動展示網站](https://wujin57.github.io/AIOT-DIC2/)**

https://wujin57.github.io/AIOT-DIC2/

免安裝，直接透過瀏覽器操作。

## Demo 畫面

以下畫面擷取自線上 Demo，展示 LASSO 自動選擇特徵與五種方法比較。

### 1. 特徵選擇與模型設定

![首頁與特徵選擇控制：LASSO 自動選擇三個特徵](docs/images/01-overview.png)

比較 **Pearson Correlation、Mutual Information、RFE、RFECV、LASSO** 五種方法。支援固定特徵數的演算法可選擇 k = 1–5；RFECV 與 LASSO 使用自動選擇的特徵數。

### 2. 方法比較與模型評估

![五種方法比較表、模型評估指標及特徵數與表現曲線](docs/images/02-comparison.png)

點選 **Compare All 5 Methods**，並排查看各方法的選取特徵、CV RMSE、Test RMSE 與 Test R²。透過驗證誤差與解釋變異曲線，觀察模型複雜度和預測表現的關係。

### 3. 模型係數與即時獲利預測

![迴歸係數、即時獲利預測、實際與預測散佈圖及殘差診斷](docs/images/03-prediction.png)

查看迴歸係數，輸入研發支出、行政支出、行銷支出及州別，即時計算預估獲利。搭配實際值與預測值散佈圖、殘差診斷圖，檢視模型誤差。

### 4. 殘差檢視與資料集探索

![測試資料殘差表與資料集概覽](docs/images/04-residuals-dataset.png)

查看 10 筆測試資料的實際獲利、預測獲利、殘差及絕對誤差，並下載殘差 CSV、原始資料與模型 JSON。資料集包含 50 筆記錄，4 個原始預測變數經編碼後形成 5 個模型欄位。

## 快速體驗

1. 開啟 [Demo 網站](https://wujin57.github.io/AIOT-DIC2/)。
2. 選擇特徵選擇演算法，調整可用的特徵數。
3. 點選 **Compare All 5 Methods** 比較結果。
4. 在 **Live profit predictor** 輸入支出與州別，觀察預估獲利。
5. 查看下方的預測圖、殘差表與資料集。

## 專案資訊

- **資料集：** [50_Startups.csv](https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv)
- **模型：** scikit-learn `LinearRegression`，特徵選擇後重新擬合 OLS。
- **評估：** 80/20 訓練測試切分、`random_state=42`、訓練資料上的 5-fold cross-validation。
- **技術：** Python、pandas、NumPy、scikit-learn、HTML、CSS、JavaScript、Chart.js。
- **部署：** GitHub Pages，推送到 `main` 後由官方 Actions 自動部署。

模型係數代表關聯而非因果；預估獲利是模型估計，並非保證。

網站操作、本機預覽與部署方式詳見 [網站使用與開發說明](docs/DEVELOPMENT.md)。
