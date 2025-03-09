const API_URL = "https://eeg-api-117396735687.asia-south2.run.app"; // Ensure correct endpoint

const sendEEGData = async (eegArray) => {
  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ data: eegArray }),
    });

    const result = await response.json();
    console.log("Mood Prediction:", result);
  } catch (error) {
    console.error("API Error:", error);
  }
};