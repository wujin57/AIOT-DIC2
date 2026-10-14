# 網站使用與開發說明

[開啟線上 Demo](https://wujin57.github.io/AIOT-DIC2/) · [返回專案首頁](../README.md)

## 網站功能與操作

| 區塊 | 操作與內容 |
| --- | --- |
| 特徵選擇 | 切換 Pearson Correlation、Mutual Information、RFE、RFECV 或 LASSO。 |
| 特徵數量 | Pearson、Mutual Information 與 RFE 可選 k = 1–5；RFECV 與 LASSO 顯示 Auto k。 |
| 方法比較 | 點選 **Compare All 5 Methods**，查看各方法的特徵、驗證誤差與測試表現。 |
| 模型評估 | 查看 RMSE、R²、MAE、特徵數與模型表現曲線。切換模型時，指標、圖表與迴歸方程式會同步更新。 |
| 即時預測 | 在 **Live profit predictor** 輸入研發、行政、行銷支出及州別，查看預估獲利。 |
| 誤差檢視 | 查看實際值與預測值散佈圖、殘差圖及測試資料的誤差明細。 |
| 資料下載 | 下載原始 CSV、目前模型的殘差 CSV，以及模型結果 JSON。 |

網站支援桌面與手機瀏覽。預測欄位需輸入非負數值；超出訓練資料範圍時會顯示提醒。

## 網站檔案

網站內容位於 `dashboard/`，使用 HTML、CSS、JavaScript 與 Chart.js。

| 檔案 | 用途 |
| --- | --- |
| [`index.html`](../dashboard/index.html) | 頁面結構與各功能區塊。 |
| [`styles.css`](../dashboard/styles.css) | 配色、卡片、表格與響應式版面。 |
| [`app.js`](../dashboard/app.js) | 模型切換、圖表更新、即時預測及下載功能。 |
| [`data/models.json`](../dashboard/data/models.json) | 網站讀取的模型參數、指標與預測資料。 |
| [`data/50_Startups.csv`](../dashboard/data/50_Startups.csv) | 提供下載的原始資料集。 |
| [`vendor/chart.umd.min.js`](../dashboard/vendor/chart.umd.min.js) | 本機隨附的 Chart.js 圖表套件。 |
| [`qa.html`](../dashboard/qa.html) | 網站互動功能的瀏覽器檢查頁面。 |

網站載入預先產生的模型結果，並在瀏覽器中計算即時預測；部署時不需要 Python 後端或 npm 建置。

## 本機預覽

在專案根目錄執行：

```sh
python3 -m http.server 8000 --directory dashboard
```

開啟 http://localhost:8000 即可預覽網站。請透過 HTTP 伺服器開啟，直接雙擊 `index.html` 可能因瀏覽器限制而無法載入 JSON。

需要檢查互動功能時，開啟 http://localhost:8000/qa.html，頁面上方會顯示 PASS 或 FAIL。

## GitHub Pages 自動部署

部署設定位於 [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)。

1. 將 GitHub 儲存庫的 **Settings → Pages → Build and deployment → Source** 設為 **GitHub Actions**。
2. 推送更新到 `main`，工作流會檢查網站必要檔案及 JSON 格式，接著上傳並部署 `dashboard/`。
3. 在 [Actions 頁面](https://github.com/wujin57/AIOT-DIC2/actions) 查看執行狀態，也可選擇 **Deploy regression dashboard → Run workflow** 手動部署。
4. 部署成功後，開啟 [Demo 網站](https://wujin57.github.io/AIOT-DIC2/) 查看更新。

## 網站設計與授權

頁面設計參考 [Multiple Linear Regression Explorer](https://huanchen1107.github.io/AIoT-DIC2-MultipleLR/)。Chart.js 採用 MIT 授權，詳見 [授權文件](../dashboard/vendor/LICENSE.chartjs.md)。
