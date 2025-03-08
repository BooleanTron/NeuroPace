import React, { useEffect, useState } from "react";
import { View, Text, Modal, ActivityIndicator, Button, Image } from "react-native";

const StressPopup = ({ visible, onClose }) => {
  const [stressData, setStressData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (visible) {
      fetch("http://your-flask-api-url/stress-popup")
        .then((response) => response.json())
        .then((data) => {
          setStressData(data);
          setLoading(false);
        })
        .catch((error) => console.error("Error fetching stress data:", error));
    }
  }, [visible]);

  return (
    <Modal visible={visible} transparent animationType="slide">
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "rgba(0,0,0,0.5)" }}>
        <View style={{ width: "90%", padding: 20, backgroundColor: "white", borderRadius: 10 }}>
          {loading ? (
            <ActivityIndicator size="large" color="#ff0000" />
          ) : (
            <>
              <Text style={{ fontSize: 24, fontWeight: "bold", color: "red" }}>{stressData.warning}</Text>
              <Text style={{ fontSize: 18, marginVertical: 10 }}>{stressData.recommendations}</Text>
              <Text style={{ fontSize: 20, fontWeight: "bold", color: "red" }}>Take a deep breath...</Text>
              <Button title="Close" onPress={onClose} color="#ff0000" />
            </>
          )}
        </View>
      </View>
    </Modal>
  );
};

const MoodPopup = ({ visible, onClose }) => {
  const [moodData, setMoodData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (visible) {
      fetch("http://your-flask-api-url/mood-popup")
        .then((response) => response.json())
        .then((data) => {
          setMoodData(data);
          setLoading(false);
        })
        .catch((error) => console.error("Error fetching mood data:", error));
    }
  }, [visible]);

  return (
    <Modal visible={visible} transparent animationType="slide">
      <View style={{ flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "rgba(0,0,0,0.5)" }}>
        <View style={{ width: "90%", padding: 20, backgroundColor: "white", borderRadius: 10, alignItems: "center" }}>
          {loading ? (
            <ActivityIndicator size="large" color="#0000ff" />
          ) : (
            <>
              <Text style={{ fontSize: 24, fontWeight: "bold" }}>{moodData.mood}</Text>
              <Image source={{ uri: moodData.emoji }} style={{ width: 80, height: 80, marginVertical: 10 }} />
              <Text style={{ fontSize: 18 }}>{moodData.description}</Text>
              <Button title="Close" onPress={onClose} color="#0000ff" />
            </>
          )}
        </View>
      </View>
    </Modal>
  );
};

export { StressPopup, MoodPopup };