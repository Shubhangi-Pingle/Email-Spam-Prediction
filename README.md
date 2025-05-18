
#  Email Spam Prediction using Machine Learning

This project focuses on building a machine learning model to classify emails as **spam** or **not spam (ham)**. It uses Natural Language Processing (NLP) techniques and machine learning algorithms to detect spam emails based on their textual content.

##  Project Overview

* Clean and preprocess email text data
* Convert text into numerical features using TF-IDF Vectorization
* Train and evaluate machine learning models
* Predict whether an email is spam or not

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
* NumPy
* Matplotlib / Seaborn (for visualization)
* Jupyter Notebook (or any Python IDE)

## 📂 Dataset

The dataset used is provided in the given folder. (e.g., 'spam' or 'ham').

Sample format:

| Label | Message                     |
| ----- | --------------------------- |
| ham   | Hey, how are you doing?     |
| spam  | Win a \$1000 gift card now! |

## 🔍 Features

* Text Preprocessing (lowercasing, punctuation removal, stopwords removal, stemming)
* Feature Extraction using TF-IDF
* Model Training using:

  * Naive Bayes
  * K-Nearest Neighbors
  * Random Forest
  * Support Vector Classifier
  * Logistic Regression
  * Decision Tree
* Evaluation using Accuracy and Precision
* Voting Classifier implementation
* Confusion Matrix and ROC Curve Visualization

## 📊 Model Evaluation

| Algorithm | Accuracy | Precision |
| --------- | -------- | --------- |
| KN        | 0.905222 | 1.000000  |
| NB        | 0.970986 | 1.000000  |
| RF        | 0.975822 | 0.982906  |
| SVC       | 0.975822 | 0.974790  |
| LR        | 0.958414 | 0.970297  |
| DT        | 0.927466 | 0.811881  |

## 🧪 Installation & Usage

1. **Clone the repository**

   ```
   git clone https://github.com/yourusername/email-spam-prediction.git
   cd email-spam-prediction
   ```

2. **Run the model**

   * Open the notebook: `sms-spam-detection.ipynb`
   * OR run the Python script:

     ```
     python spam_detector.py
     ```

## 📄 License

This project is licensed under the MIT License.

---

