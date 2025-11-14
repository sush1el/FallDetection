/**
 * CAIretaker Fall Detection System - ESP32 Buzzer Alert Module
 * 
 * This code connects to your Flask backend and monitors for fall incidents.
 * When a fall is detected, it activates a buzzer alarm that sounds continuously
 * until the incident status returns to normal.
 * 
 * Hardware Setup:
 * - ESP32 GPIO23 → Buzzer Positive (+) [or through transistor for 5V buzzer]
 * - ESP32 GND → Buzzer Negative (-)
 * 
 * Author: RAGAS - Pamantasan ng Lungsod ng Maynila
 * Project: CAIretaker AI-Based Fall Detection System
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// ========================================
// CONFIGURATION - UPDATE THESE VALUES
// ========================================

// WiFi Credentials
const char* WIFI_SSID = "PLDTHOMEFIBR2A1A3";           // Replace with your WiFi name
const char* WIFI_PASSWORD = "PLDTWIFI1HMBT";   // Replace with your WiFi password

// Flask Backend Configuration
const char* BACKEND_URL = "http://192.168.1.31:5000";  // Replace with your Flask server IP address

// Hardware Configuration 
const int BUZZER_PIN = 2;  // Changed from 23 to 2 for ESP32-CAM

// Timing Configuration
const unsigned long CHECK_INTERVAL = 500;      // Check backend every 500ms
const unsigned long WIFI_RETRY_DELAY = 5000;   // Retry WiFi every 5 seconds

// ========================================
// GLOBAL VARIABLES
// ========================================

bool fallDetected = false;
bool previousFallState = false;
unsigned long lastCheckTime = 0;
unsigned long wifiRetryTime = 0;
bool wifiConnected = false;

HTTPClient http;
WiFiClient client;

// ========================================
// SETUP - RUNS ONCE
// ========================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n╔══════════════════════════════════════════════════════╗");
  Serial.println("║   CAIretaker Fall Detection - Buzzer Alert System   ║");
  Serial.println("║        Pamantasan ng Lungsod ng Maynila             ║");
  Serial.println("╚══════════════════════════════════════════════════════╝\n");
  
  // Initialize hardware
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
  Serial.println("✓ Buzzer initialized on GPIO23");
  
  // Connect to WiFi
  connectToWiFi();
  
  Serial.println("🔄 Starting continuous monitoring...\n");
}

// ========================================
// MAIN LOOP - RUNS CONTINUOUSLY
// ========================================

void loop() {
  unsigned long currentTime = millis();
  
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    wifiConnected = false;
    if (currentTime - wifiRetryTime >= WIFI_RETRY_DELAY) {
      Serial.println("⚠ WiFi disconnected. Reconnecting...");
      connectToWiFi();
      wifiRetryTime = currentTime;
    }
    return;
  }
  
  wifiConnected = true;
  
  // Check backend status at regular intervals
  if (currentTime - lastCheckTime >= CHECK_INTERVAL) {
    lastCheckTime = currentTime;
    
    // Fetch fall detection status
    if (fetchFallStatus()) {
      updateBuzzer();
    }
  }
  
  delay(10);
}

// ========================================
// WIFI FUNCTIONS
// ========================================

void connectToWiFi() {
  Serial.println("Connecting to WiFi...");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    wifiConnected = true;
    Serial.println("\n✓ WiFi Connected!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    
    // Test buzzer on connection
    testBuzzer();
  } else {
    wifiConnected = false;
    Serial.println("\n✗ WiFi Connection Failed!");
  }
}

void testBuzzer() {
  Serial.println("Testing buzzer...");
  for(int i = 0; i < 3; i++) {
    digitalWrite(BUZZER_PIN, HIGH);
    delay(200);
    digitalWrite(BUZZER_PIN, LOW);
    delay(200);
  }
  Serial.println("✓ Buzzer test complete");
}

// ========================================
// BACKEND COMMUNICATION
// ========================================

bool fetchFallStatus() {
  String statusUrl = String(BACKEND_URL) + "/status";
  
  http.begin(client, statusUrl);
  http.setTimeout(3000);
  int httpCode = http.GET();
  
  if (httpCode == 200) {
    String payload = http.getString();
    http.end();
    
    DynamicJsonDocument doc(4096);
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) {
      Serial.println("✗ JSON parsing failed");
      return false;
    }
    
    // Check for fall incidents
    JsonArray detections = doc["detections"].as<JsonArray>();
    int fallCount = 0;
    
    for (JsonObject detection : detections) {
      bool isFall = detection["is_fall"] | false;
      if (isFall) {
        fallCount++;
        int personId = detection["id"] | 0;
        int incidentId = detection["incident_id"] | 0;
        Serial.print("🚨 FALL DETECTED - Person ID: ");
        Serial.print(personId);
        Serial.print(", Incident #");
        Serial.println(incidentId);
      }
    }
    
    // Update fall state
    fallDetected = (fallCount > 0);
    return true;
    
  } else {
    http.end();
    Serial.print("✗ HTTP Error: ");
    Serial.println(httpCode);
    return false;
  }
}

// ========================================
// BUZZER CONTROL
// ========================================

void updateBuzzer() {
  // Fall just detected - activate buzzer
  if (fallDetected && !previousFallState) {
    Serial.println("\n🚨🚨🚨 FALL ALERT ACTIVATED 🚨🚨🚨");
    digitalWrite(BUZZER_PIN, HIGH);
    Serial.println("Buzzer: ON (Continuous)");
  } 
  // Fall cleared - deactivate buzzer
  else if (!fallDetected && previousFallState) {
    Serial.println("\n✓✓✓ FALL ALERT CLEARED ✓✓✓");
    digitalWrite(BUZZER_PIN, LOW);
    Serial.println("Buzzer: OFF");
  }
  
  previousFallState = fallDetected;
}
