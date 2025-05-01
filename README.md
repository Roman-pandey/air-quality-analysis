# air-quality-analysis
Data analysis and visualization of air quality using Python
# Air Quality Data Analysis and Visualization

This project performs an in-depth analysis of air quality data using Python. It covers data preprocessing, statistical analysis, and a variety of visualizations to understand pollution patterns, geographical trends, and pollutant behaviors over time.

## 📂 Dataset
The dataset used is named `Air_Quality.csv`, and it contains air pollution records with attributes such as:
- `Start_Date`: Date of data recording
- `Name`: Pollutant name
- `Data Value`: Measured value of pollutant
- `Geo Place Name`: City/Location
- `Geo Type Name`: Type of geographic area
- Other relevant metadata

> Note: Make sure `Air_Quality.csv` is placed in the same directory as `main.py`.

## 🧪 Features and Objectives

This project includes:
- 🧼 **Data Cleaning and Preprocessing**
- 📊 **Exploratory Data Analysis (EDA)** including:
  - Missing value detection
  - Summary statistics
- 📈 **Visualization Objectives**:
  1. Line chart of top 3 pollutants over years
  2. Top 10 most polluted cities by average value
  3. NO₂ concentration in top 10 locations
  4. Outlier detection using scatter plot
  5. Histogram of pollution values
  6. Boxplot by pollutant
  7. Heatmap of correlation matrix
  8. Count plot per pollutant
  9. Pollution variation by month
  10. Pairwise relationships between features
  11. Heatmap of pollutant trends by year
  12. Pollution levels by geo type
  13. Geo type distribution (pie chart)

## 📦 Requirements

Install the required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn
