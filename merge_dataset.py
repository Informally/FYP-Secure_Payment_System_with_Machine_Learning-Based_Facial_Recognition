import pandas as pd

casia_test_csv = r'C:/Users/User/Desktop/Dataset/test_labels.csv'
replay_test_csv = r'C:/Users/User/Desktop/Dataset/another_dataset/samples/replay_test_labels.csv'
merged_test_csv = r'C:/Users/User/Desktop/Dataset/merged_test_labels.csv'

casia_test = pd.read_csv(casia_test_csv)
replay_test = pd.read_csv(replay_test_csv)
replay_test = replay_test.rename(columns={'folder': 'subject'})  # if needed

merged_test = pd.concat([casia_test, replay_test], ignore_index=True)
merged_test.to_csv(merged_test_csv, index=False)
print(f"Merged test CSV saved to {merged_test_csv}")