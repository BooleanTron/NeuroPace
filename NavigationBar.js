import React from "react";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import { NavigationContainer } from "@react-navigation/native";
import { MaterialIcons } from "@expo/vector-icons";

// Importing Screens
import HomeScreen from "./HomeScreen";
import StatisticsPage from "./StatisticsPage";
import MentalHealthPage from "./MentalHealthPage";
import ConsultDoctorPage from "./ConsultDoctorPage";

const Tab = createBottomTabNavigator();

const NavigationBar = () => {
  return (
    <NavigationContainer>
      <Tab.Navigator
        screenOptions={({ route }) => ({
          tabBarIcon: ({ color, size }) => {
            let iconName;
            if (route.name === "Home") {
              iconName = "home";
            } else if (route.name === "Statistics") {
              iconName = "bar-chart";
            } else if (route.name === "Mental Health") {
              iconName = "self-improvement";
            } else if (route.name === "Consult") {
              iconName = "chat";
            }
            return <MaterialIcons name={iconName} size={size} color={color} />;
          },
          tabBarActiveTintColor: "#6200ee",
          tabBarInactiveTintColor: "gray",
        })}
      >
        <Tab.Screen name="Home" component={HomeScreen} />
        <Tab.Screen name="Statistics" component={StatisticsPage} />
        <Tab.Screen name="Mental Health" component={MentalHealthPage} />
        <Tab.Screen name="Consult" component={ConsultDoctorPage} />
      </Tab.Navigator>
    </NavigationContainer>
  );
};

export default NavigationBar;
