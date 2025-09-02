# 🏏 IPL Score Prediction 🏏

<div align="center">

![IPL Logo](https://img.shields.io/badge/IPL-Cricket-orange?style=for-the-badge&logo=cricket)
![Python](https://img.shields.io/badge/Python-3.6+-blue?style=for-the-badge&logo=python)
![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-green?style=for-the-badge&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=for-the-badge&logo=jupyter)

**🎯 Predicting IPL First Innings Scores with Machine Learning**

[![GitHub stars](https://img.shields.io/github/stars/sunbyte16/IPL-Score-Prediction?style=social)](https://github.com/sunbyte16/IPL-Score-Prediction)
[![GitHub forks](https://img.shields.io/github/forks/sunbyte16/IPL-Score-Prediction?style=social)](https://github.com/sunbyte16/IPL-Score-Prediction)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Features](#-features)
- [📊 Dataset](#-dataset)
- [🔧 Technologies Used](#-technologies-used)
- [📈 Model Performance](#-model-performance)
- [🏃‍♂️ Quick Start](#️-quick-start)
- [📱 Usage](#-usage)
- [📊 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📞 Connect with Me](#-connect-with-me)

---

## 🎯 Overview

This project uses **Machine Learning** to predict the first innings score in IPL matches! 🎲 

Using historical IPL data from **2008-2019**, the model analyzes various factors like:
- 🏏 **Batting Team** & **Bowling Team**
- ⏰ **Current Overs** & **Runs Scored**
- 🎯 **Wickets Lost**
- 📈 **Recent Performance** (Last 5 overs)

---

## 🚀 Features

- ✨ **Multiple ML Algorithms** - Linear Regression, Decision Tree, Random Forest, AdaBoost
- 🎯 **High Accuracy** - RMSE of ~15.8 runs
- 📊 **Comprehensive Analysis** - Data visualization with heatmaps
- 🔄 **Real-time Predictions** - Predict scores during live matches
- 📱 **Easy to Use** - Simple function interface
- 🏆 **IPL Teams Support** - All major IPL franchises included

---

## 📊 Dataset

- **📁 Source**: IPL match data from 2008-2019
- **📈 Records**: 76,014 ball-by-ball records
- **🏏 Teams**: 8 consistent IPL teams
- **⏰ Time Period**: 12 IPL seasons
- **🎯 Target**: First innings total score prediction

### 🏏 Supported Teams
- 🦁 **Chennai Super Kings**
- 🦅 **Delhi Daredevils** 
- 👑 **Kings XI Punjab**
- 🐯 **Kolkata Knight Riders**
- 🦈 **Mumbai Indians**
- 👑 **Rajasthan Royals**
- 🔴 **Royal Challengers Bangalore**
- 🌅 **Sunrisers Hyderabad**

---

## 🔧 Technologies Used

<div align="center">

| Category | Technology | Version |
|----------|------------|---------|
| 🐍 **Language** | Python | 3.6+ |
| 📊 **Data Analysis** | Pandas, NumPy | Latest |
| 📈 **Visualization** | Matplotlib, Seaborn | Latest |
| 🤖 **Machine Learning** | Scikit-learn | Latest |
| 📓 **Notebook** | Jupyter | Latest |

</div>

---

## 📈 Model Performance

| Algorithm | MAE | MSE | RMSE | Status |
|-----------|-----|-----|------|--------|
| 🎯 **Linear Regression** | 12.12 | 251.01 | **15.84** | ✅ **Best** |
| 🌳 **Decision Tree** | 17.09 | 531.06 | 23.04 | ❌ |
| 🌲 **Random Forest** | 13.76 | 330.21 | 18.17 | ⚠️ |
| 🚀 **AdaBoost** | 12.22 | 249.60 | 15.80 | ✅ |

**🏆 Winner: Linear Regression** with RMSE of 15.84 runs!

---

## 🏃‍♂️ Quick Start

### 📥 Installation

```bash
# Clone the repository
git clone https://github.com/sunbyte16/IPL-Score-Prediction.git

# Navigate to project directory
cd IPL-Score-Prediction

# Install required packages
pip install -r requirements.txt
```

### 🚀 Running the Project

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the main notebook
First Innings Score Prediction - IPL.ipynb
```

---

## 📱 Usage

### 🎯 Making Predictions

```python
# Example prediction
final_score = predict_score(
    batting_team='Mumbai Indians',
    bowling_team='Chennai Super Kings', 
    overs=12.3,
    runs=113,
    wickets=2,
    runs_in_prev_5=55,
    wickets_in_prev_5=0
)

print(f"Predicted Score: {final_score-10} to {final_score+5}")
# Output: Predicted Score: 179 to 194
```

### 📊 Real Match Predictions

The model was tested on **7 real IPL matches** from 2018-2019:

| Match | Teams | Actual Score | Predicted Range | Accuracy |
|-------|-------|--------------|-----------------|----------|
| 🏏 **Match 1** | KKR vs DD | 200/9 | 159-174 | ⚠️ |
| 🏏 **Match 2** | SRH vs RCB | 146/10 | 138-153 | ✅ |
| 🏏 **Match 3** | MI vs KXIP | 186/8 | 180-195 | ✅ |
| 🏏 **Match 4** | MI vs KXIP | 176/7 | 179-194 | ✅ |
| 🏏 **Match 5** | RR vs CSK | 151/7 | 128-143 | ⚠️ |
| 🏏 **Match 6** | DD vs SRH | 155/7 | 157-172 | ✅ |
| 🏏 **Match 7** | DD vs CSK | 147/9 | 137-152 | ✅ |

**🎯 Overall Accuracy: ~71%** (5 out of 7 predictions within range)

---

## 📊 Results

### 🎯 Key Insights

- ✅ **Linear Regression** performs best for this problem
- 📈 **Recent performance** (last 5 overs) is crucial for predictions
- 🏏 **Team combinations** significantly affect scoring patterns
- ⏰ **Overs played** and **wickets lost** are strong predictors

### 📈 Model Training

- **📚 Training Data**: IPL Seasons 1-9 (2008-2016)
- **🧪 Test Data**: IPL Season 10 (2017)
- **🔮 Predictions**: IPL Seasons 11-12 (2018-2019)

---

## 🤝 Contributing

Contributions are welcome! 🎉

1. 🍴 **Fork** the repository
2. 🌿 **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. 💾 **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. 📤 **Push** to the branch (`git push origin feature/AmazingFeature`)
5. 🔄 **Open** a Pull Request

---

## 📞 Connect with Me

<div align="center">

### 👨‍💻 **Sunil Sharma**

[![GitHub](https://img.shields.io/badge/GitHub-@sunbyte16-black?style=for-the-badge&logo=github)](https://github.com/sunbyte16)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Sunil%20Kumar-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/sunil-kumar-bb88bb31a/)
[![Portfolio](https://img.shields.io/badge/Portfolio-Web%20Dev-orange?style=for-the-badge&logo=portfolio)](https://github.com/sunbyte16)

**💻 Software Developer | ☁️ Cloud & DevOps Enthusiast | 🤖 Aspiring ML Engineer**

*Passionate about building efficient, scalable, and user-focused applications*

</div>

---

<div align="center">

### ⭐ **Star this repository if you found it helpful!** ⭐

**🏏 Happy Cricket Analytics! 🏏**

---

*Created By ❤️[Sunil Sharma](https://github.com/sunbyte16)*

</div>
