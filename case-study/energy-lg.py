import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
import holidays
from lightgbm import early_stopping, LGBMRegressor
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

# Load the dataset
df_or = pd.read_csv('/Users/melisacevik/Desktop/time-series-master/datasets/PJMW_hourly.csv')

# Convert 'Datetime' to proper datetime format
df_or["Datetime"] = pd.to_datetime(df_or["Datetime"], format="%Y-%m-%d %H:%M:%S")

# Sort data and reset the index
df = df_or.sort_values(by="Datetime").reset_index(drop=True)

# Adding public holidays as a feature (U.S. holidays)
us_holidays = holidays.US()
df['is_holiday'] = df['Datetime'].apply(lambda x: 1 if x in us_holidays else 0)


# Date-related features
def date_features(df, col):
    df_c = df.copy()
    df_c['month'] = df_c[col].dt.month
    df_c['hour'] = df_c[col].dt.hour
    df_c['quarter'] = df_c[col].dt.quarter
    df_c['dayofweek'] = df_c[col].dt.dayofweek
    df_c['dayofmonth'] = df_c[col].dt.day
    df_c['month_sin'] = np.sin(df_c.month * (2. * np.pi / 12))
    df_c['month_cos'] = np.cos(df_c.month * (2. * np.pi / 12))
    df_c['hour_sin'] = np.sin(df_c.hour * (2. * np.pi / 24))
    df_c['hour_cos'] = np.cos(df_c.hour * (2. * np.pi / 24))
    df_c['week_sin'] = np.sin(df_c.dayofweek * (2. * np.pi / 7))
    df_c['week_cos'] = np.cos(df_c.dayofweek * (2. * np.pi / 7))
    return df_c


df = date_features(df, "Datetime")

# Drop unnecessary columns
df.drop(["month", "hour", "dayofweek", "dayofmonth"], axis=1, inplace=True)

# Split data into train and test sets (last 1 year as test)
last_date = df["Datetime"].max()
one_year_before = last_date - pd.DateOffset(years=1)

train = df[df["Datetime"] < one_year_before]
test = df[df["Datetime"] >= one_year_before]

# Test setinin minimum tarihini bul
test_min_date = test["Datetime"].min()

# Train setinin sonu ve test setinin başlangıcı arasında 24 saatlik kaydırma işlemi yap
shift_boundary = test_min_date - pd.Timedelta(hours=24)

# Add lag features for 'PJMW_MW' - Test setinin başlangıcına göre kaydırma yapılacak
df['PJMW_MW_lag1'] = df['PJMW_MW'].shift(24)
df['PJMW_MW_lag2'] = df['PJMW_MW'].shift(48)  # Lag by 1 day (24 hours)
df['PJMW_MW_lag3'] = df['PJMW_MW'].shift(72)


# Rolling işlemi yapılacak sütunlar ve boundary baz alınarak kaydırma yapılıyor
def rol_col(df, shift_boundary):
    cols = df.select_dtypes(include=[np.number]).columns
    df_rolling = pd.DataFrame(index=df.index)

    for col in cols:
        df_rolling[col + '_rol1'] = df[col].shift(1)
        df_rolling[col + '_rol24'] = df[col].shift(24)

        # Rolling işlemi, shift_boundary'ye göre uygulanacak
        df_rolling.loc[df["Datetime"] >= shift_boundary, col + '_rol24'] = np.nan

    return df_rolling


# Rolling mean for past 7 days (168 hours)
df['PJMW_MW_rolling_mean_7d'] = df['PJMW_MW'].rolling(window=168).mean()

# Log transformation of 'PJMW_MW'
df['PJMW_MW_log'] = np.log1p(df['PJMW_MW'])

# Drop rows with NaN values due to lagging/rolling
df.dropna(inplace=True)

# Convert 'quarter' to categorical
df.info()
df["quarter"] = df["quarter"].astype("object")
df["is_holiday"] = df["is_holiday"].astype("object")

df = pd.get_dummies(df, columns=["is_holiday", "quarter"], drop_first=True)

# Split the data again after rolling and lagging
train = df[df["Datetime"] < one_year_before]
test = df[df["Datetime"] >= one_year_before]

# Target variable 'PJMW_MW'
X_train = train.drop("PJMW_MW", axis=1)
y_train = train["PJMW_MW"]

X_test = test.drop("PJMW_MW", axis=1)
y_test = test["PJMW_MW"]

# Drop 'Datetime' column from features
X_train.drop("Datetime", axis=1, inplace=True)
X_test.drop("Datetime", axis=1, inplace=True)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define and train the model
model = LGBMRegressor(random_state=42, n_estimators=1000)
model.fit(X_train_scaled, y_train,
          eval_set=[(X_test_scaled, y_test)],
          eval_metric='mae',
          callbacks=[early_stopping(stopping_rounds=10)])

# Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"R²: {r2}")

# Correlation matrix visualization
corr_matrix = df.corr()
fig = go.Figure(data=go.Heatmap(
    z=corr_matrix.values,
    x=corr_matrix.columns,
    y=corr_matrix.columns,
    colorscale='Viridis',
    text=corr_matrix.values,
    hoverinfo='text'
))
fig.update_layout(title='Correlation Matrix', xaxis_nticks=36, width=900, height=900)
fig.show()

# Save the correlation matrix to a CSV file
corr_matrix.to_csv("correlation_matrix.csv")

