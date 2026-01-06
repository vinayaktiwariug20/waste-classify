# ♻️ WasteVision: AI-Powered Image Classification

## 📌 Project Overview
This project began as a comparative study of deep learning backbones for a computer vision assignment. While the initial goal was to evaluate model performance on the RealWaste dataset, I took it a step further by transitioning the model from a static Google Colab environment into a functional, interactive web application.

By deploying the model via Streamlit, I transformed a theoretical exercise into a practical tool that can classify arbitrary waste images in real-time.

## 🚀 The "Colab to Production" Journey
Most machine learning models begin and end in a notebook. My goal was to ensure this model was accessible and testable.
* Rapid Prototyping: Leveraged AI-assisted development to build a robust Streamlit frontend and deployment pipeline.
* Model Selection: Evaluated four different CNN backbones provided in the curriculum. EfficientNet was selected as the final production model because it strictly outperformed the other candidates across all key metrics (F1-Score and AUC).
* Real-World Testing: The live demo allows users to bypass pre-cleaned datasets and test the model against "in-the-wild" images.

## 📊 Performance Metrics
The model was trained on the RealWaste dataset and achieved the following results:
* F1-Score: 0.861
* AUC: 0.988 (Demonstrating superior class separation compared to other tested architectures)

## 🛠️ Tech Stack
* Deep Learning: EfficientNet (Backbone)
* Dataset: RealWaste
* Interface: Streamlit (Web UI)
* Deployment: Streamlit Cloud / GitHub

## 💻 How to Run This Locally

1. Clone the repository:
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

2. Install dependencies:
pip install -r requirements.txt

3. Launch the app:
streamlit run app.py

## 🎥 Demo & Resources
* Live Demo: https://waste-classify-vt.streamlit.app/
* Video Demo: https://www.youtube.com/watch?v=hJXq16BBI3c
