#Fake News Detection using Machine Learning
🚀 Detecting Fake News with KNN & Naïve Bayes

📌 Project Overview
This project aims to classify news articles as real or fake using machine learning algorithms. The model is trained on a dataset containing fake and real news articles and predicts whether a given news article is trustworthy or not.

⚡ Features
✔️ Fake vs. real news classification
✔️ Implementation of K-Nearest Neighbors (KNN) and Naïve Bayes
✔️ GUI-based application for user interaction
✔️ Model training and evaluation
✔️ Dataset preprocessing and feature extraction

🛠 Tech Stack
Programming Language: Python
Machine Learning: Scikit-Learn (KNN, Naïve Bayes)
Dataset: Fake and real news dataset
GUI: Tkinter / PyQt / Flask (whichever you used)
Other Tools: Pandas, NumPy, Pickle

📁 Fake-News-Detection  
│── 📂 dataset/             # Contains fake and real news datasets  
│── 📂 model/               # Trained models (saved as .pkl files)  
│── 📂 gui/                 # GUI-based application files  
│── 📜 main.py              # Main script for model training/testing  
│── 📜 train_model.py       # Script to train the model  
│── 📜 predict.py           # Script for making predictions  
│── 📜 requirements.txt     # Required libraries  
│── 📜 README.md            # Project documentation  

🚀 How to Run the Project
```git clone https://github.com/yourusername/fake-news-detection.git```
```cd fake-news-detection```

```pip install -r requirements.txt```

```python train_model.py```

```python main.py```

📊 Model Performance
Algorithm	Accuracy
Naïve Bayes	85%
KNN	80%
🔥 Future Improvements
✅ Implement deep learning models (LSTM, BERT)
✅ Deploy the model as a web app
✅ Improve dataset preprocessing
