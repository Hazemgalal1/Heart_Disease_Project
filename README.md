# Comprehensive Machine Learning Pipeline for Heart Disease Prediction

A full end-to-end Machine Learning pipeline built on the **UCI Heart Disease dataset**, covering everything from **data preprocessing** to **deployment**.  
This project demonstrates practical ML techniques including dimensionality reduction, feature selection, model training, hyperparameter tuning, and deployment with **Streamlit**.

---

## Features
- Complete ML pipeline on real-world medical data.
- Preprocessing & feature engineering.
- Supervised & unsupervised learning experiments.
- Multiple model training and evaluation.
- Final model deployment with an interactive Streamlit app.
- Ready-to-use project structure for reproducibility.

---

## Live Demo
Try the deployed app here:  
👉 [Heart Disease Prediction App](https://e318baa2d519.ngrok-free.app/)

---

## Project Structure
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

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Hazemgalal1/heart-disease-ml-pipeline.git
cd heart-disease-ml-pipeline
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Locally
```bash
streamlit run ui/app.py
```
App will be available at: `http://localhost:8501`

### 4. Run on Google Colab (with ngrok)
1. Upload `ui/app.py` and `models/final_model.pkl`
2. Install dependencies:
   ```bash
   !pip install streamlit pyngrok pandas numpy scikit-learn joblib
   ```
3. Start app with ngrok tunnel:
   ```python
   from pyngrok import ngrok
   ngrok.set_auth_token("YOUR_NGROK_TOKEN")
   get_ipython().system_raw('streamlit run app.py --server.port 8501 &')
   public_url = ngrok.connect(8501)
   print(public_url.public_url)
   ```

---

## Models Trained
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- K-Nearest Neighbors (KNN)  

**Final Model**: Random Forest (best performance overall)

---

## Results
- Detailed evaluation metrics: [results/evaluation_metrics.txt](results/evaluation_metrics.txt)  
- Full report: [reports/Final_Report.pdf](reports/Final_Report.pdf)

---

## Requirements
See [requirements.txt](requirements.txt) for the full list of dependencies.  

Main libraries:
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Pyngrok (for Colab deployment)

---

## Future Improvements
- Add deep learning experiments (TensorFlow / PyTorch).
- Explore model interpretability with SHAP/LIme.
- Improve UI design with advanced Streamlit components.
- Deploy to a cloud platform (AWS/GCP/Heroku).

---

## Author
**Hazem Galal Abd Elsatar**  
- GitHub: [@Hazemgalal1](https://github.com/Hazemgalal1)  
- LinkedIn: [Hazem Galal](https://www.linkedin.com/in/hazem-galal-439356226)  

---

## License
This project is licensed under the MIT License.
