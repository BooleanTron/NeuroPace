import React, { useState, useEffect } from "react";
import { View, Text, ActivityIndicator, ScrollView } from "react-native";
import { ProgressBar } from "react-native-paper";
import { LineChart } from "react-native-chart-kit";
import { Dimensions } from "react-native";

const StatisticsScreen = () => {
  const [monthlyData, setMonthlyData] = useState([]);
  const [averageMood, setAverageMood] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch("http://your-flask-api-url/statistics")
      .then((response) => response.json())
      .then((data) => {
        setMonthlyData(data.mood_scores);
        setAverageMood(data.average_mood);
        setLoading(false);
      })
      .catch((error) => console.error("Error fetching statistics:", error));
  }, []);

  return (
    <ScrollView contentContainerStyle={{ alignItems: "center", padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: "bold" }}>Monthly Statistics</Text>
      {loading ? (
        <ActivityIndicator size="large" color="#6200ee" />
      ) : (
        <>
          <Text style={{ fontSize: 18, marginVertical: 10 }}>Average Mood Score: {averageMood}/10</Text>
          <ProgressBar progress={averageMood / 10} color="#6200ee" style={{ width: 200, height: 10, marginBottom: 20 }} />
          <Text style={{ fontSize: 18, marginBottom: 10 }}>Mood Score Over Time</Text>
          <LineChart
            data={{
              labels: monthlyData.map((_, index) => index + 1),
              datasets: [{ data: monthlyData }],
            }}
            width={Dimensions.get("window").width - 40}
            height={220}
            yAxisLabel=""
            yAxisSuffix="/10"
            chartConfig={{
              backgroundColor: "#ffffff",
              backgroundGradientFrom: "#ffffff",
              backgroundGradientTo: "#ffffff",
              decimalPlaces: 1,
              color: (opacity = 1) => `rgba(98, 0, 238, ${opacity})`,
              labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
              style: {
                borderRadius: 16,
              },
              propsForDots: {
                r: "6",
                strokeWidth: "2",
                stroke: "#6200ee",
              },
            }}
            bezier
            style={{ marginVertical: 8, borderRadius: 16 }}
          />
        </>
      )}
    </ScrollView>
  );
};

export default StatisticsScreen;