#include <WiFi.h>
#include <HTTPClient.h>
#include "driver/i2s.h"
#include <ArduinoJson.h>

// WiFi Config
const char* ssid     = "SSID";
const char* password = "PASSWORD";

// FastAPI Endpoint
String serverURL_chunk = "http://[YOUR_IP]:5000/upload/audio_chunk";

// I2S Microphone (INMP441)
#define I2S_WS      15
#define I2S_SCK     14
#define I2S_SD      32
#define I2S_PORT    I2S_NUM_0

void i2s_install() {
    const i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = 16000,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = (i2s_comm_format_t)(I2S_COMM_FORMAT_I2S),
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 256,
        .use_apll = false
    };
    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);

    const i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK,
        .ws_io_num = I2S_WS,
        .data_out_num = -1,
        .data_in_num = I2S_SD
    };
    i2s_set_pin(I2S_PORT, &pin_config);
    i2s_zero_dma_buffer(I2S_PORT);
}

// LED + Buzzer
#define RED_PIN    25
#define GREEN_PIN  26
#define YELLOW_PIN 27
#define BUZZER_PIN 33

void setStatus(String status) {
    if (status == "safe") {
        digitalWrite(GREEN_PIN, HIGH);
        digitalWrite(YELLOW_PIN, LOW);
        digitalWrite(RED_PIN, LOW);
        noTone(BUZZER_PIN);
    }
    else if (status == "suspicious") {
        digitalWrite(GREEN_PIN, LOW);
        digitalWrite(YELLOW_PIN, HIGH);
        digitalWrite(RED_PIN, LOW);
        noTone(BUZZER_PIN);
    }
    else if (status == "alert") {
        digitalWrite(GREEN_PIN, LOW);
        digitalWrite(YELLOW_PIN, LOW);
        digitalWrite(RED_PIN, HIGH);
        tone(BUZZER_PIN, 2000);
    }
}

// Setup
void setup() {
    Serial.begin(115200);

    pinMode(RED_PIN, OUTPUT);
    pinMode(GREEN_PIN, OUTPUT);
    pinMode(YELLOW_PIN, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);

    i2s_install();

    WiFi.begin(ssid, password);
    Serial.print("Connecting");
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nConnected!");
}

// Loop
#define BUFFER_SIZE 1024
uint8_t audioBuffer[BUFFER_SIZE];

void loop() {
    size_t bytes_read;
    i2s_read(I2S_PORT, (void*)audioBuffer, sizeof(audioBuffer), &bytes_read, portMAX_DELAY);

    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(serverURL_chunk);
        http.addHeader("Content-Type", "application/octet-stream");
        
        int code = http.POST(audioBuffer, bytes_read);

        if (code > 0) {
            String response = http.getString();
            Serial.println("AI Response: " + response);

            StaticJsonDocument<200> doc;
            if (!deserializeJson(doc, response)) {
                String status = doc["status"];
                setStatus(status);
            }
        } else {
            Serial.println("Failed to send audio chunk");
        }

        http.end();
    }

    delay(100);  // 10 chunks per second
}