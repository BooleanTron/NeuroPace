import React, { useEffect, useState } from "react";
import { View, Text, Modal, ActivityIndicator, Button } from "react-native";

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

export default StressPopup;
