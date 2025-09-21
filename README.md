# ❤️ Comprehensive Machine Learning Full Pipeline on Heart Disease UCI Dataset

## 📌 Overview
This project applies a full machine learning pipeline on the UCI Heart Disease dataset.  
It includes preprocessing, dimensionality reduction (PCA), feature selection, supervised & unsupervised learning, hyperparameter tuning, and a Streamlit app for deployment.

---

## 🌍 Live Demo
You can try the Streamlit app here:  
👉 [Heart Disease Prediction App](https://e318baa2d519.ngrok-free.app/)

---

## 📂 Project Structure
```
Heart_Disease_Project/
│── data/                       # Raw & processed datasets
│   ├── heart_disease.csv
│   ├── train.csv
│   ├── test.csv
│
│── models/                     # Trained ML models
│   ├── final_model.pkl
│
│── ui/                         # Streamlit app
│   ├── app.py
│
│── reports/                    # Reports for each phase
│   ├── Final_Report.pdf
│
│── deployment/                 # Deployment setup
│   ├── ngrok_setup.txt
│
│── results/                    # Evaluation metrics
│   ├── evaluation_metrics.txt
│
│── requirements.txt            # Project dependencies
│── README.md                   # Documentation
│── .gitignore
```

---

## 🚀 How to Run the Streamlit App

### 🔹 Local (VS Code / Terminal)
```bash
pip install -r requirements.txt
streamlit run ui/app.py
```
Open `http://localhost:8501` in your browser.

---

### 🔹 Google Colab (with ngrok)
1. Upload `ui/app.py` and `models/final_model.pkl` to Colab:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```
2. Install dependencies:
   ```bash
   !pip install streamlit pyngrok pandas numpy scikit-learn joblib
   ```
3. Set up ngrok:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   ```
4. Run:
   ```python
   get_ipython().system_raw('streamlit run app.py --server.port 8501 &')
   public_url = ngrok.connect(8501)
   print(public_url.public_url)
   ```

---

## 📊 Models Trained
- Logistic Regression
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

✅ **Final Selected Model**: Random Forest (best performance)

---

## 📈 Results
See [results/evaluation_metrics.txt](results/evaluation_metrics.txt) for model evaluation metrics.  
See [reports/Final_Report.pdf](reports/Final_Report.pdf) for the full documentation.

---

## 🛠 Requirements
See [requirements.txt](requirements.txt) for the full list of dependencies.

---
