# **FinCrypt 2.0**

A powerful dual function crypto platform combining **Bitcoin price prediction** and **live crypto news aggregation** inside a modern Streamlit dashboard.

---

## **Overview**

FinCrypt 2.0 integrates machine-learning forecasting with real-time crypto news:

- Support Vector Regression (SVR) BTC predictions  
- Custom CSV upload (`Date`, `Price`)  
- Interactive charting + forecast table  
- Live crypto news feed  
- Coin logo dropdown  
- Search bar for filtering news  
- Dark/Light mode for themed cards  
- Multi-language support (incl. Hindi)  
- Clean two-tab interface: **Price Forecast** & **Crypto News**

---

## **Features**

### **Bitcoin Price Prediction**
- SVR model using TimeSeriesSplit  
- Forecast range: 1–30 days  
- Option to upload your own dataset  
- Best parameter display + RMSE  
- Forecast visualization (Matplotlib)  
- Table with formatted future prices  

---

### **Live Crypto News Dashboard**
- Integrates **NewsData.io Crypto API**  
- Emoji coin dropdown (BTC, ETH, SOL, etc.)  
- Languages: EN, ES, DE, FR, IT, PT, **HI**  
- Dark/Light mode switch  
- Search bar  
- Two-column card UI  
- Metadata display: source + published date  

---

## **Tabs**

### **Price Forecast**
- Dataset preview  
- Model training and metrics  
- Forecast table  
- Combined historical + predicted chart  

### **Crypto News**
- Live feed  
- Search filtering  
- Responsive cards  
- Themed UI (Dark/Light)  

---

## **Installation**

### 1. Clone the repository
```bash
git clone https://github.com/yourname/FinCrypt-2.0.git
cd FinCrypt-2.0/ui

pip install -r ../requirements.txt

# for running in terminal
python -m streamlit run app.py

The app will open at:
http://localhost:8501
```

## Crypto News API Setup

You need a free API key from **NewsData.io**:

1. Visit: https://newsdata.io  
2. Create a free account  
3. Go to **API Keys** → Copy your key  
4. Paste into the app’s sidebar:

NewsData.io API key → YOUR_KEY_HERE

---

## Supported Coins (with logos)

| Coin          | Symbol |
|---------------|--------|
| 🟧 Bitcoin     | BTC    |
| 🟪 Ethereum    | ETH    |
| 🟦 Solana      | SOL    |
| 💠 Ripple      | XRP    |
| 🟨 Binance Coin| BNB    |
| 🐕 Dogecoin    | DOGE   |
| 🟥 Cardano     | ADA    |
| ⚪ Litecoin    | LTC    |
| 🎯 Polkadot    | DOT    |
| 🧊 Avalanche   | AVAX   |

---

## CSV Format Example

Your uploaded CSV should follow this format:

|Date |Price |
|2021-01-01 | 29200 |
|2021-01-02 | 32100 |

---

## Supported Languages

| Language   | Code |
|------------|------|
| English    | en   |
| Spanish    | es   |
| German     | de   |
| French     | fr   |
| Italian    | it   |
| Portuguese | pt   |
| Hindi      | hi   |

---

## UI Highlights

### **Prediction Tab**
- Clean visual layout  
- Centered metrics  
- Modern chart  
- Auto-formatting tables  
- Clean typography  

### **News Tab**
- Two-column responsive card layout  
- Dark/light toggle  
- Search bar for real-time filtering  
- Minimalistic aesthetic  

---

## Example Forecast Chart
*(Chart displays dynamically in the app)*

---

## Machine Learning Details

- **Model:** Support Vector Regression (SVR)  
- **Kernel:** RBF  
- **Hyperparameter search:** GridSearchCV  
- **CV Method:** TimeSeriesSplit (5 folds)  
- **Evaluation Metric:** RMSE  

---

## Future Enhancements
- Add more crypto models (ETH, SOL)  
- Add LSTM-based deep learning forecasting  
- Add more news providers  
- Add sentiment analysis for news  

---

## Contributing

Pull requests welcome.  
For major changes, open an issue first to discuss what you would like to add.

---

## License

This project is licensed under the **MIT License**.

---

## Support

If you found this helpful, please **star the repository!**

---

## Thank You

Thank you for using the **Bitcoin Price Predictor + Crypto News Dashboard**!

---
