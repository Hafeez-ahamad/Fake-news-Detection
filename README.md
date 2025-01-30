# Fake News Detection System

This project focuses on detecting fake news articles using a **Naïve Bayes** machine learning model. The system preprocesses textual data using **TF-IDF** and achieves a **precision score of 90%** in identifying fake news. It also includes a **user-friendly interface** for testing real-time news inputs.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [File Structure](#file-structure)
3. [Setup Instructions](#setup-instructions)
4. [Usage](#usage)
5. [Model Training](#model-training)
6. [Making Predictions](#making-predictions)
7. [User Interface](#user-interface)
8. [Results](#results)
9. [Contributing](#contributing)
10. [License](#license)

---

## Project Overview
The goal of this project is to classify news articles as **real** or **fake** using machine learning. Key features include:
- **Data Preprocessing**: Tokenization, text cleaning, and TF-IDF vectorization.
- **Model**: Naïve Bayes classifier.
- **Performance**: Achieves **90% precision** in detecting fake news.
- **User Interface**: A Flask-based web interface for real-time predictions.

---

## File Structure

---

## Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/fake-news-detection.git
   cd fake-news-detection
pip install -r requirements.txt
import nltk
nltk.download('punkt')
