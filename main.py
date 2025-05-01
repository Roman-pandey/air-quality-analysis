import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
#======= DATA PREPROCESSING & CLEANING ======

df = pd.read_csv("Air_Quality.csv")#Load dataset
df = df.drop(columns=['Message'])# Drop the 'Message' column as it's empty
df["Data Value"] = df["Data Value"].round(2)# Change "Data Value" column values upto two decimal places:

df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')# Convert 'Start_Date' to datetime
df['Year'] = df['Start_Date'].dt.year# Extract year from 'Start_Date'
grouped = df.groupby(['Name', 'Year'])['Data Value'].mean().reset_index()# Group by pollutant name and year to get average data value

# ======= EDA(EXPLORATORY DATA ANALYSIS) ======

print("Missing Values ==>>>")
print(df.isnull().sum())
print("\nDataset Overview ==>>>")
print(df.head())
print("\nDataset Info ==>>>")
print(df.info())
print("\n Summary Statistics ==>>>")
print(df.describe(include='all'))
print("\n")


# ====== CENTRAL TENDENCY & DISPERSION ======
print("Mean,Median,Mode,Range,Variance,Standard Deviation of 'Data Value' column ==>>>\n")
print("Mean:", df['Data Value'].mean())
print("Median:", df['Data Value'].median())
print("Mode:", df['Data Value'].mode()[0])
print("Range:", df['Data Value'].max() - df['Data Value'].min())
print("Variance:", df['Data Value'].var())
print("Standard Deviation:", df['Data Value'].std())


#OBJECTIVE 1:-> Average of top 3 most frequently recorded pollutants across multiple years - (Seaborn line chart);
top_pollutants = df['Name'].value_counts().head(3).index
filtered_grouped = grouped[grouped['Name'].isin(top_pollutants)]

plt.figure(figsize=(12, 6))
sns.lineplot(data=filtered_grouped, x='Year', y='Data Value', hue='Name', marker='o')
plt.title('Average Pollutant Levels Over Years')
plt.ylabel('Average Data Value')
plt.xlabel('Year')
plt.grid(True)
plt.tight_layout()
plt.show()


#OBJECTIVE 2:-> Average Pollution Levels – Top 10 Locations - (Seaborn horizontal bar chart);
top_locations = df['Geo Place Name'].value_counts().head(10).index
avg_pollution_location = df[df['Geo Place Name'].isin(top_locations)].groupby('Geo Place Name')['Data Value'].mean().sort_values()
plt.figure(figsize=(12, 6))
bars = sns.barplot(x=avg_pollution_location.values, y=avg_pollution_location.index, palette="viridis")
plt.title("Average Pollution Levels - Top 10 Locations")
plt.xlabel("Average Data Value")
plt.ylabel("Location")
# Add data labels to each bar
for i, (value, name) in enumerate(zip(avg_pollution_location.values, avg_pollution_location.index)):
    plt.text(value + 0.5, i, f"{value:.1f}", va='center')

plt.tight_layout()
plt.show()


#OBJECTIVE 3 :-> top 10 most polluted areas based on the average concentration of Nitrogen Dioxide (NO₂) levels
top_polluted = (
    df.groupby("Geo Place Name")["Data Value"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
)
plt.figure(figsize=(12, 6))
bars = plt.bar(top_polluted.index, top_polluted.values, color='steelblue')
plt.title('Top 10 Most Polluted Cities (by Average NO2 Levels)')
plt.ylabel('Average NO2 (ppb)')
plt.xlabel('City')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval:.2f}', ha='center', va='bottom')

plt.show()


#OBJECTIVE 4:-> Pollution Levels Over the Years (Outlier Detection) - (Seaborn scatter plot);
plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Year', y='Data Value', hue='Name', alpha=0.6, palette='tab10')
plt.title("Pollution Levels Over the Years (Outlier Detection)")
plt.xlabel("Year")
plt.ylabel("Data Value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


#OBJECTIVE 5:-> Distribution of Pollution Data Values (Histogram)
plt.figure(figsize=(10, 6))
sns.histplot(df['Data Value'], bins=50, kde=True, color='skyblue')
plt.title("Distribution of Pollution Data Values")
plt.xlabel("Data Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


#OBJECTIVE 6:-> This boxplot visualizes the distribution of pollution levels (data values) for the top 5 most frequently recorded pollutants.
plt.figure(figsize=(14, 6))
top_pollutants = df['Name'].value_counts().head(5).index
filtered_df = df[df['Name'].isin(top_pollutants)]
sns.boxplot(data=filtered_df, x='Name', y='Data Value', palette='Set2')
plt.title("Boxplot of Pollution Levels by Pollutant")
plt.xlabel("Pollutant")
plt.ylabel("Data Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# OBJECTIVE 7:-> This heatmap visualizes the correlation coefficients between all numerical variables in the dataset.
plt.figure(figsize=(8, 5))
sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()


# OBJECTIVE 8:->  visualize the number of records for each type of pollutant in the dataset
plt.figure(figsize=(12, 6))
ax = sns.countplot(data=df, y='Name', order=df['Name'].value_counts().index, palette='coolwarm')
plt.title("Number of Records per Pollutant Type")
plt.xlabel("Count")
plt.ylabel("Pollutant")
# Add count labels
for p in ax.patches:
    count = int(p.get_width())
    ax.text(count + 1, p.get_y() + p.get_height() / 2, count, va='center')

plt.tight_layout()
plt.show()


# OBJECTIVE 9 :-> visualize how average pollution levels vary across the months of the year
df['Month'] = df['Start_Date'].dt.month
monthly_avg = df.groupby('Month')['Data Value'].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.lineplot(data=monthly_avg, x='Month', y='Data Value', marker='o')
plt.title("Average Pollution Level by Month")
plt.xlabel("Month")
plt.ylabel("Average Data Value")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()
plt.show()


# OBJECTIVE 10 :-> To show pairwise relationships between numerical features in the dataset.
sns.pairplot(df[['Year', 'Geo Join ID', 'Data Value']], diag_kind='kde')
plt.suptitle("Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.show()


# OBJECTIVE 11 :-> visualize how the average levels of different pollutants have changed over the years.
pollutant_year = df.groupby(['Name', 'Year'])['Data Value'].mean().unstack().fillna(0)
plt.figure(figsize=(14, 6))
sns.heatmap(pollutant_year, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Average Pollutant Levels by Year")
plt.xlabel("Year")
plt.ylabel("Pollutant")
plt.tight_layout()
plt.show()


# OBJECTIVE 12 :-> To show the average pollution level by geographic area type 
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=df, x='Geo Type Name', y='Data Value', estimator='mean', ci=None, palette='cubehelix')
plt.title("Average Pollution by Geo Type")
plt.xlabel("Geo Type")
plt.ylabel("Average Data Value")
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x() + p.get_width() / 2, height + 0.5, f'{height:.1f}', ha='center')

plt.tight_layout()
plt.show()


# OBJECTIVE 13 :-> visualize the distribution of the top 5 most common geographic types in dataset.
geo_type_counts = df['Geo Type Name'].value_counts().head(5)
plt.figure(figsize=(8, 8))
plt.pie(geo_type_counts, labels=geo_type_counts.index, autopct='%1.1f%%', startangle=140)
plt.title("Top 5 Geo Type Distribution")
plt.tight_layout()
plt.show()





