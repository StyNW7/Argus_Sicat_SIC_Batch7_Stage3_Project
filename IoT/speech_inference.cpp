#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "YOUR_WIFI";
const char* password = "YOUR_PASSWORD";

const char* serverURL = "http://192.168.1.10:5000/upload"; // Python server URL

// === MICROPHONE ===
// Gunakan ADC pin (contoh GPIO 34)
int micPin = 34;
int sampleRate = 16000;  // 16 kHz
int seconds = 1;
int totalSamples = sampleRate * seconds;

int16_t audioBuffer[20000]; // 20k sample buffer

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
  }
  Serial.println("WiFi Connected.");
}

void loop() {
  Serial.println("Recording...");

  // Record PCM audio
  for (int i = 0; i < totalSamples; i++) {
    audioBuffer[i] = analogRead(micPin);
    delayMicroseconds(62);  // ~16 kHz sampling
  }

  // Convert PCM to WAV (simple header + data)
  uint8_t wavData[totalSamples * 2 + 44];
  createWavHeader(wavData, totalSamples);

  memcpy(wavData + 44, audioBuffer, totalSamples * 2);

  // HTTP POST
  HTTPClient http;
  http.begin(serverURL);
  http.addHeader("Content-Type", "audio/wav");

  int resp = http.POST(wavData, sizeof(wavData));

  Serial.printf("Upload status: %d\n", resp);
  http.end();

  delay(2000);  // record every 2 secs
}

// WAV Header generator
void createWavHeader(uint8_t* header, int samples) {
  int fileSize = 44 + samples * 2;
  int byteRate = 16000 * 2;

  memcpy(header, "RIFF", 4);
  *(int*)(header + 4) = fileSize - 8;
  memcpy(header + 8, "WAVEfmt ", 8);
  *(int*)(header + 16) = 16;  // Subchunk1Size
  *(short*)(header + 20) = 1; // PCM
  *(short*)(header + 22) = 1; // Mono
  *(int*)(header + 24) = 16000; 
  *(int*)(header + 28) = byteRate;
  *(short*)(header + 32) = 2; // block align
  *(short*)(header + 34) = 16; // bits per sample
  memcpy(header + 36, "data", 4);
  *(int*)(header + 40) = samples * 2;
}
