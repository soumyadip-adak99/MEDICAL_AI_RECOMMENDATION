# 🏥 Medical AI Recommendation System

## 📋 Project Overview

The **Medical AI Recommendation System** is an intelligent web application designed to connect patients with the most suitable medical specialists. Using Machine Learning (SVC Model), the system analyzes user-provided symptoms, predicts possible diseases, and recommends doctors based on specialization, experience, and ratings.

This project aims to reduce self-diagnosis anxiety and make expert medical help easily accessible.

---

## ✨ Key Features

* **🩺 AI-Powered Disease Prediction** – Uses a trained Support Vector Classifier for accurate disease prediction.
* **👨‍⚕️ Smart Doctor Recommendation** – Suggests the best doctors based on specialization and patient needs.
* **⚖️ Dynamic Sorting** – Sort recommendations by Experience, Rating, or Balanced Score.
* **🖥️ Modern UI** – Clean and responsive medical dashboard with user-friendly navigation.
* **🔍 Smart Symptom Search** – Handles varied symptom inputs intelligently.

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn (SVC Model), NumPy
* **Data Processing:** Pandas
* **Frontend:** HTML5, CSS3, JavaScript (Jinja2)
* **Dataset:** Custom dataset mapping symptoms → diseases → doctors

---

## 📂 Project Structure

```
MEDICAL_AI_RECOMMENDATION/
├── data/
│   └── Doctor_data_with_diseases.csv
├── model/
│   └── svc.pkl
├── static/
├── templates/
│   └── index.html
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

Follow the steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/soumyadip-adak99/MEDICAL_AI_RECOMMENDATION.git
cd MEDICAL_AI_RECOMMENDATION
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If requirements.txt is missing:

```bash
pip install flask pandas numpy scikit-learn
```

### 4. Run the Application

```bash
python app.py
```

### 5. Access in Browser

Visit:

```
http://127.0.0.1:5000/
```

---

## 💡 How to Use

1. Enter symptoms (comma-separated).
2. Choose sorting preference (Experience / Rating / Both).
3. Click **Analyze & Recommend**.
4. View predicted disease.
5. See top recommended doctors with details.

---

## 📊 Model & Data

* **Algorithm:** Support Vector Classifier (SVC)
* **Training:** Symptom–disease dataset using vectorized bag-of-words
* **Doctor Recommendation:** Matches predicted disease with doctor specialties from CSV dataset

---

## 📜 License

This project is open-source under the **MIT License**.

---

<div align="center">
<b>Developed by <a href="https://github.com/soumyadip-adak99">Soumyadip Adak</a></b>
</div>
