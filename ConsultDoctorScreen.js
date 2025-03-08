import React, { useState } from "react";
import { View, Text, TextInput, Button, ScrollView } from "react-native";

const ConsultDoctorScreen = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = { text: inputText, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInputText("");

    try {
      const response = await fetch("http://your-flask-api-url/consult", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: inputText }),
      });
      const data = await response.json();

      const botMessage = { text: data.reply, sender: "bot" };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Error fetching response:", error);
    }
  };

  return (
    <View style={{ flex: 1, padding: 20 }}>
      <Text style={{ fontSize: 24, fontWeight: "bold", marginBottom: 10 }}>Consult with a Doctor</Text>
      <ScrollView style={{ flex: 1, marginBottom: 10 }}>
        {messages.map((msg, index) => (
          <View
            key={index}
            style={{
              alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
              backgroundColor: msg.sender === "user" ? "#6200ee" : "#f1f1f1",
              padding: 10,
              borderRadius: 10,
              marginVertical: 5,
            }}
          >
            <Text style={{ color: msg.sender === "user" ? "#fff" : "#000" }}>{msg.text}</Text>
          </View>
        ))}
      </ScrollView>
      <TextInput
        style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 10, marginBottom: 10 }}
        placeholder="Ask a doctor..."
        value={inputText}
        onChangeText={setInputText}
      />
      <Button title="Send" onPress={sendMessage} color="#6200ee" />
    </View>
  );
};

export default ConsultDoctorScreen;
