import pandas as pd

# Load the CSV file, explicitly handling the first row as header
df = pd.read_csv('coordinates.csv', header=None)

# Add a new column with the value 0
df['new_column'] = 0

# Save the modified DataFrame back to a CSV file
df.to_csv('path_to_your_modified_file.csv', index=False, header=False)

print("A new column has been added and the file has been saved.")
