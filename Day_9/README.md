# 📘 Understanding Data in Computing

This repository contains lecture notes and an interactive Jupyter Notebook (`data.ipynb`) for **Day 9: Understanding Data in Computing**.  
It is designed for students to **learn, explore, and practice** core concepts of data representation, analysis, and visualization in Python.

---

## 🚀 Topics Covered

### 1. Introduction
- Data as raw, unorganized facts and figures.
- Data vs Information.

---

### 2. Structured vs. Unstructured Data
- **Structured Data**: Organized, tabular (e.g., SQL, Excel).  
- **Unstructured Data**: Text, images, audio, videos, IoT sensor logs.

#### 🔑 Key Differences

| Feature       | Structured Data                        | Unstructured Data                 |
|---------------|-----------------------------------------|------------------------------------|
| Organization  | Fixed schema (tables)                  | No predefined format               |
| Type          | Quantitative (numbers, dates)          | Qualitative (text, media)          |
| Ease of Query | Easy with SQL                          | Complex; requires NLP, AI, vision  |
| Storage       | Relational DBs, Data Warehouses        | Data Lakes, NoSQL, File Systems    |
| Flexibility   | Less flexible                          | Highly flexible                    |

---

### 3. Quantitative vs. Qualitative Data

- **Quantitative Data (📊)**: Numerical, measurable.  
  Examples: Temperature = 25°C, Height = 175cm, Students = 30  
- **Qualitative Data (📝)**: Descriptive, categorical.  
  Examples: Car color = “Blue”, Feeling = “Happy”, Eye color = “Brown”  

#### 🔑 Comparison

| Feature     | Quantitative Data         | Qualitative Data        |
|-------------|----------------------------|-------------------------|
| Type        | Numbers, measurable        | Descriptions, labels    |
| Questions   | How many? How much?        | Why? How?               |
| Analysis    | Statistical, charts        | Thematic, interpretation|
| Form        | Numbers, tables, graphs    | Text, narratives, images|
| Objectivity | Objective                  | Subjective              |

---

### 4. Numerical Variables
- **Continuous Variables (📏)**: Infinite values in range (Height, Weight, Temperature).  
- **Discrete Variables (🔢)**: Countable, whole numbers only (Dice rolls, Students in class).  

| Feature  | Continuous Variable | Discrete Variable |
|----------|----------------------|-------------------|
| Values   | Any value in a range | Specific whole numbers |
| Nature   | Measurable           | Countable |
| Examples | Height, Time         | Dice rolls, Class size |

---

### 5. Qualitative (Categorical) Variables
- **Nominal**: Categories with no order (Eye color, Country, Pets).  
- **Ordinal**: Categories with ranking (Satisfaction level, Education).  
- **Binary**: Two categories only (Yes/No, Pass/Fail).  

| Feature  | Nominal | Ordinal | Binary |
|----------|----------|---------|--------|
| Meaning  | No order | Ordered categories | Two outcomes only |
| Example  | Eye color | Education level | Pass/Fail |

---

### 6. Measures of Central Tendency
- **Mean**: Arithmetic average (sensitive to outliers).  
- **Median**: Middle value (resistant to outliers).  
- **Mode**: Most frequent value (works with categorical data).  

---

### 7. Measures of Spread
- **Range** = Max − Min  
- **Variance** = Avg squared deviation from mean  
- **Standard Deviation (σ)** = Typical distance from mean  
- **Interquartile Range (IQR)** = Spread of middle 50%  

---

### 8. Shape of Data Distribution
- **Symmetrical (Normal)**: Mean = Median = Mode.  
- **Right Skew (Positive)**: Mean > Median.  
- **Left Skew (Negative)**: Mean < Median.  
- **Uniform**: All values equally likely.  
- **Bimodal**: Two peaks (mixed populations).  

---

## 🐍 Jupyter Notebook (`data.ipynb`)

The notebook includes:
- **Python examples** for each concept.  
- **Visualizations**: Histograms, bar charts, boxplots, and synthetic distributions.  
- **Step-by-step explanations** with both numeric and categorical examples.  
- **Helper functions** for computing summary statistics.

### 📊 Sample Visualizations
- Histogram of temperatures (quantitative data).  
- Bar chart of categorical variables (eye colors, pass/fail).  
- Boxplot showing spread & IQR.  
- Distributions: Normal, Uniform, Skewed, and Bimodal.  

---

## 📝 Assignments

### Assignment 1
1. Load a CSV and classify as **structured or unstructured**.  
2. Compute mean, median, mode, variance, and std for a numeric column.  
3. Visualize using histogram + boxplot.  

### Assignment 2
1. Collect 50 short text messages and find most common words.  
2. Plot bar chart of top 10 words.  
3. Cross-tabulate two categorical columns and visualize as grouped bar chart.  

### Assignment 3
1. Generate a **bimodal dataset** by mixing distributions.  
2. Fit a KDE (density plot) and identify the two peaks.  
3. Use a real dataset (e.g., salaries) to compute **skewness** and justify whether mean or median is more representative.  

---

## 📂 Repository Contents
- `Day_9_Notes.pdf` → Lecture notes in PDF format.  
- `data.ipynb` → Interactive Jupyter Notebook with examples and visualizations.  
- `README.md` → This file.  

---

## 🎯 How to Use
1. Clone the repository.  
2. Open `data.ipynb` in **Jupyter Notebook / JupyterLab**.  
3. Run cells step by step to learn and visualize concepts.  
4. Attempt assignments at the end for practice.  

---

## 👨‍🏫 Author
Prepared for educational purposes to help students **understand and apply fundamental data concepts** interactively.

---

