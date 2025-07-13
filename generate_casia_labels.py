import pandas as pd

# Path to your labels.csv
labels_csv = r'C:/Users/User/Desktop/Dataset/another_dataset/samples/labels.csv'

# Load the CSV
df = pd.read_csv(labels_csv)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split
split_idx = int(0.8 * len(df))
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

# Save
train_df.to_csv(r'C:/Users/User/Desktop/Dataset/another_dataset/samples/replay_train_labels.csv', index=False)
test_df.to_csv(r'C:/Users/User/Desktop/Dataset/another_dataset/samples/replay_test_labels.csv', index=False)

print("Train and test splits created!")