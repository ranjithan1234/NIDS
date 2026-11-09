# Network Intrusion Detection System (NIDS)

A machine learning-based Network Intrusion Detection System (NIDS) designed to identify potential network attacks and classify network traffic as normal or malicious.

## 📌 Project Overview

Network security is an important aspect of modern computing. Traditional security systems may struggle to identify evolving network threats.

This project uses machine learning techniques to analyze network traffic and detect potential intrusions. It provides a pipeline for data preprocessing, model training, prediction, and an optional web-based interface.

## 🎯 Objectives

- Detect potential network intrusions using machine learning.
- Analyze network traffic datasets.
- Preprocess data for machine learning models.
- Train and evaluate classification models.
- Predict the nature of network traffic.
- Provide an optional interactive interface for users.

## ✨ Features

- Network traffic preprocessing
- Categorical feature encoding
- Numerical feature scaling
- Machine learning model training
- Prediction on new CSV datasets
- Model and preprocessing artifact storage
- Optional Streamlit-based interface

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Machine Learning | Network intrusion detection |
| Scikit-learn | Data preprocessing and ML algorithms |
| Pandas | Dataset manipulation |
| NumPy | Numerical operations |
| Streamlit | Interactive web interface |
| Joblib | Saving and loading trained models |

Additional dependencies are listed in `requirements.txt`.

## 📂 Dataset

This project uses the **NSL-KDD dataset**, a widely used benchmark dataset for network intrusion detection research.

The dataset contains network traffic records used for training and testing machine learning models.

Dataset directory:

`data/NSL-KDD-Dataset-master/`

## 🧠 Machine Learning Models

The project includes implementations or support for the following models:

- Decision Tree
- K-Nearest Neighbors (KNN)
- XGBoost
- LightGBM

The models are used to learn patterns from network traffic data and identify potential intrusions.

## 📁 Project Structure

```text
NIDS/
│
├── app.py
├── train_ids.py
├── predict.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   └── NSL-KDD-Dataset-master/
│
├── model_output/
│
└── outputs/
    └── models/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ranjithan1234/NIDS.git
```

### 2. Navigate to the project directory

```bash
cd NIDS
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / macOS:**

```bash
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

## 🚀 Running the Project

### Step 1: Train the machine learning models

```bash
python train_ids.py
```

This executes the model training pipeline.

### Step 2: Run predictions

```bash
python predict.py --input new_data.csv --output outputs/predictions/preds.csv
```

Replace `new_data.csv` with the path to your input dataset.

### Step 3: Launch the web interface

```bash
streamlit run app.py
```

This launches the optional Streamlit interface in your browser.

## 🔍 Applications

- Network security monitoring
- Intrusion detection research
- Machine learning-based traffic classification
- Cybersecurity learning and experimentation

## 🔮 Future Enhancements

- Improve detection of previously unseen attacks.
- Add real-time network traffic monitoring.
- Enhance model performance through hyperparameter tuning.
- Develop a more interactive dashboard.
- Integrate additional machine learning and deep learning models.

## 👩‍💻 Author

**Ranjitha N N**

GitHub: [ranjithan1234](https://github.com/ranjithan1234)

## 📌 Repository

[Network Intrusion Detection System – GitHub](https://github.com/ranjithan1234/NIDS)

---

*This project was developed for educational and research purposes in network security and machine learning.*
