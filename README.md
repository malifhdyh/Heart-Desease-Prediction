# 🫀 Heart Disease Prediction

This project aims to predict the likelihood of heart disease using **Machine Learning** techniques and an interactive **Streamlit** application.  
Developed collaboratively by **Muhammad Alif Hidayah** and **Alfriando**.  

---

## 📌 Project Overview
Cardiovascular diseases are among the leading causes of death worldwide.  
By leveraging data science, we can build predictive models that help in early detection of heart disease risks.  

In this project, we:  
- Performed **data preprocessing** and exploratory data analysis (EDA).  
- Trained multiple **machine learning models** (Logistic Regression, Decision Tree, Random Forest).  
- Selected the best-performing model and saved it as a `.sav` file.  
- Built an **interactive Streamlit web app** for real-time predictions.  

---

## 📂 Repository Structure
```

├── app.py                         # Streamlit web application
├── final\_pipe.sav                 # Trained ML model (pipeline)
├── heart\_disease\_prediction.ipynb # Jupyter Notebook (EDA & training)
├── heart\_disease\_uci.csv          # Dataset used
├── requirements.txt               # Project dependencies
├── runtime.txt                    # Runtime environment configuration
└── README.md                      # Project documentation

````

---

## ⚙️ Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/malifhdyh/Heart-Desease-Prediction.git
cd Heart-Desease-Prediction
````

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

---

## 📊 Dataset

We used the **UCI Heart Disease dataset**, which contains various health indicators such as:

* Age
* Sex
* Chest Pain Type
* Resting Blood Pressure
* Cholesterol
* Maximum Heart Rate Achieved
* ST Depression (Oldpeak)
* Slope of Peak Exercise ST Segment

---

## 🔍 Results & Insights

* Several models were tested; **Logistic Regression** and **Random Forest** performed strongly.
* Key predictors of heart disease risk included **cholesterol levels** and **resting blood pressure**.
* The deployed Streamlit app allows users to input health data and instantly check prediction results.

---

## 🌐 Live Demo

https://heart-desease-prediction-pwd-al.streamlit.app/

---

## 👥 Authors

* **Muhammad Alif Hidayah** – Data Science & ML
* **Alfriando** – Data Analysis & Implementation

---

## ⭐ Support

If you find this project helpful, consider giving it a ⭐ on [GitHub](https://github.com/malifhdyh/Heart-Desease-Prediction)!
