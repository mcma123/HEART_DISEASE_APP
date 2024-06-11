

# Di
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset with correct delimiter
csv_file_path = r"C:\Users\HP22\Desktop\ITDAa project\heart (1).csv"
df = pd.read_csv(csv_file_path, delimiter=';')


# Columns to be one-hot encoded
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# Perform one-hot encoding
data = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Ensure all boolean columns are converted to integers
data = data.astype(int)

# Display the first few rows to confirm encoding
print(data.head())

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# The data is now ready for fitting to a machine learning model
