import pandas as pd
import requests
import json

# Load the CSV data
data = pd.read_csv('synthetic_eeg_data.csv')

# Convert the data to a list of dictionaries (matching the format used in your JS code)
eeg_array = data.to_dict(orient='records')

# Define the API URL
API_URL = "https://eeg-api-o7bc4f3mba-em.a.run.app"

# Function to send EEG data to the API and print the result
def send_eeg_data(eeg_array):
    try:
        # Send POST request to the API
        response = requests.post(API_URL, json={"data": eeg_array}, headers={"Content-Type": "application/json"})

        # If the request is successful
        if response.status_code == 200:
            result = response.json()
            print("Mood Prediction:", result)
        else:
            print(f"Error: {response.status_code}, {response.text}")
    except Exception as e:
        print("API Error:", e)

# Call the function with the EEG data
send_eeg_data(eeg_array)
