#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// WiFi Config
const char* ssid     = "SSID";
const char* password = "PASSWORD";

// FastAPI Endpoint
String serverURL_frame = "http://[YOUR_IP]:5000/upload/frame";

// Camera Config (AI Thinker ESP32-CAM)
void initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = 5;
  config.pin_d1       = 18;
  config.pin_d2       = 19;
  config.pin_d3       = 21;
  config.pin_d4       = 36;
  config.pin_d5       = 39;
  config.pin_d6       = 34;
  config.pin_d7       = 35;
  config.pin_xclk     = 0;
  config.pin_pclk     = 22;
  config.pin_vsync    = 25;
  config.pin_href     = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn     = 32;
  config.pin_reset    = -1;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Performance Settings
  config.frame_size   = FRAMESIZE_QVGA;  // 320x240
  config.jpeg_quality = 12;              
  config.fb_count     = 1;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("Camera init failed!");
    delay(2000);
    ESP.restart();
  }

  Serial.println("Camera initialized");
}

// WiFi Init
void initWiFi() {
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED) {
    delay(400);
    Serial.print(".");
    retry++;

    if (retry > 25) {
      Serial.println("\nWiFi failed, restarting...");
      ESP.restart();
    }
  }

  Serial.println("\nWiFi Connected!");
  Serial.print("IP: ");
  Serial.println(WiFi.localIP());
}

// Send Frame to FastAPI
void sendFrame(camera_fb_t* fb) {
  if (WiFi.status() != WL_CONNECTED) return;

  HTTPClient http;
  http.begin(serverURL_frame);
  http.addHeader("Content-Type", "application/octet-stream");

  int code = http.POST(fb->buf, fb->len);
  Serial.print("Upload frame -> Status: ");
  Serial.println(code);

  http.end();
}

// Main Setup
void setup() {
  Serial.begin(115200);
  initWiFi();
  initCamera();
}

// Main Loop
void loop() {
  camera_fb_t* fb = esp_camera_fb_get();

  if (!fb) {
    Serial.println("Camera capture failed");
    delay(200);
    return;
  }

  sendFrame(fb);
  esp_camera_fb_return(fb);

  delay(300);  // 3 fps
}
