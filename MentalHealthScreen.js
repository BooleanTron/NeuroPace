import React, { useState, useEffect } from "react";
import { View, Text, ActivityIndicator, ScrollView } from "react-native";
import { ProgressBar } from "react-native-paper";

const MentalHealthScreen = () => {
  const [mentalHealthScore, setMentalHealthScore] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://your-flask-api-url/mental-health")
      .then((response) => response.json())
      .then((data) => {
        setMentalHealthScore(data.mental_health_score);
        setRecommendations(data.recommendations);
        setLoading(false);
      })
      .catch((error) => console.error("Error fetching mental health data:", error));
  }, []);

  return (
    <ScrollView contentContainerStyle={{ alignItems: "center", padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: "bold" }}>Mental Health Score</Text>
      {loading ? (
        <ActivityIndicator size="large" color="#6200ee" />
      ) : (
        <>
          <Text style={{ fontSize: 18, marginVertical: 10 }}>Score: {mentalHealthScore}/10</Text>
          <ProgressBar progress={mentalHealthScore / 10} color="#6200ee" style={{ width: 200, height: 10, marginBottom: 20 }} />
          <Text style={{ fontSize: 18, marginBottom: 10 }}>Recommendations</Text>
          {recommendations.map((rec, index) => (
            <Text key={index} style={{ fontSize: 16, marginTop: 5 }}>
              - {rec}
            </Text>
          ))}
        </>
      )}
    </ScrollView>
  );
};

export default MentalHealthScreen;
