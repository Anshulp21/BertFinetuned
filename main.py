import requests
import pandas as pd

# API endpoint and authorization token
api_url = "https://bitwerx"
headers = {
    "Authorization": "Bearer JEOnSC3EeeOXU",
    "Content-Type": "application/json",
}

# Read CSV file
csv_file_path = "Mapping Test Results 11-07-2023 (1).csv"
df = pd.read_csv(csv_file_path)

# Initialize a new column for the API response
df["api_response"] = ""

# Iterate over rows and send API requests
for index, row in df.iterrows():
    transaction_descriptions = [row["transaction_description"]]
    data = {"data": transaction_descriptions}
    
    # Send POST request
    response = requests.post(api_url, json=data, headers=headers)
    api_response = response.json()

    # Update the "api_response" column
    df.at[index, "api_response"] = api_response

# Write the updated DataFrame to a new CSV file
new_csv_file_path = "csvfileall.csv"
df.to_csv(new_csv_file_path, index=False)

print(f"New CSV file created at: {new_csv_file_path}")
