import React, { useState, useEffect } from "react";
import { View, Text, ActivityIndicator } from "react-native";
import { ProgressBar } from "react-native-paper";

const HomeScreen = () => {
  const [moodScore, setMoodScore] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://your-flask-api-url/mood")
      .then((response) => response.json())
      .then((data) => {
        setMoodScore(data.mood_score);
        setRecommendations(data.recommendations);
        setLoading(false);
      })
      .catch((error) => console.error("Error fetching mood data:", error));
  }, []);

  return (
    <View style={{ flex: 1, justifyContent: "center", alignItems: "center" }}>
      <Text style={{ fontSize: 24, fontWeight: "bold" }}>Mood Score</Text>
      {loading ? (
        <ActivityIndicator size="large" color="#6200ee" />
      ) : (
        <>
          <Text style={{ fontSize: 20, marginVertical: 10 }}>{moodScore} / 10</Text>
          <ProgressBar progress={moodScore / 10} color="#6200ee" style={{ width: 200, height: 10 }} />
          <Text style={{ fontSize: 18, marginTop: 20 }}>GPT Recommendations</Text>
          {recommendations.map((rec, index) => (
            <Text key={index} style={{ fontSize: 16, marginTop: 5 }}>
              - {rec}
            </Text>
          ))}
        </>
      )}
    </View>
  );
};

export default HomeScreen;
