# IoT Service

Folder **IoT/** berisi source code untuk dua perangkat ESP32 yang digunakan dalam sistem deteksi kecurangan berbasis audio dan video.

---

## 📁 File Daftar Isi
| File | Deskripsi |
|------|------------|
| **esp32_mic_led_buzzer.ino** | Kode untuk ESP32 DevKit yang membaca audio dari INMP441, mengirim ke FastAPI, dan menyalakan LED/Buzzer berdasarkan status AI. |
| **esp32_cam.ino** | Kode untuk ESP32-CAM (AI Thinker) yang menangkap frame kamera dan mengirimkannya ke FastAPI. |

---

# 🎤 esp32_mic_led_buzzer.ino  
ESP32 DevKit berfungsi sebagai node **audio monitoring** menggunakan mikrofon digital **INMP441** melalui protokol **I2S**.

### 🔧 Fitur Utama
- Mengambil audio real-time (16 KHz, 16-bit).
- Mengirim data per 100 ms ke endpoint FastAPI `/upload/audio_chunk`.
- Menerima status dari AI: `safe`, `suspicious`, `alert`.
- Mengontrol LED (Hijau / Kuning / Merah) dan buzzer sesuai status.

### 📌 Pin Mapping — INMP441  
| INMP441 | ESP32 |
|--------|--------|
| WS | GPIO 15 |
| SCK | GPIO 14 |
| SD | GPIO 32 |
| VDD | 3.3V |
| GND | GND |

### 📌 Pin Mapping — LED & Buzzer  
| Komponen | GPIO |
|---------|------|
| LED Merah | 25 |
| LED Hijau | 26 |
| LED Kuning | 27 |
| Buzzer | 33 |

Endpoint:
[http://[YOUR_IP]:5000/upload/audio_chunk](http://[YOUR_IP]:5000/upload/audio_chunk)

---

# 📸 esp32_cam.ino  
ESP32-CAM mengirimkan **frame JPEG** secara berkala (3 FPS) ke FastAPI untuk analisis visual.

### 🔧 Fitur Utama
- Menginisialisasi kamera ESP32-CAM berbasis modul **AI Thinker**.
- Menangkap gambar dalam format JPEG QVGA (320x240).
- Mengirim setiap frame ke endpoint FastAPI `/upload/frame`.
- Restart otomatis jika WiFi gagal.

### 📌 Konfigurasi Kamera (AI Thinker)
Pin telah di-set sesuai layout resmi AI Thinker:

| Signal | GPIO |
|--------|------|
| D0 | 5 |
| D1 | 18 |
| D2 | 19 |
| D3 | 21 |
| D4 | 36 |
| D5 | 39 |
| D6 | 34 |
| D7 | 35 |
| XCLK | 0 |
| PCLK | 22 |
| VSYNC | 25 |
| HREF | 23 |
| SDA | 26 |
| SCL | 27 |
| PWDN | 32 |

Endpoint:
[http://[YOUR_IP]:5000/upload/frame](http://[YOUR_IP]:5000/upload/frame)

---

# ⚙️ Cara Menggunakan
1. Edit WiFi SSID dan password pada kedua file `.ino`.
2. Ganti `[YOUR_IP]` dengan IP laptop tempat FastAPI berjalan.
3. Upload file ke masing-masing board:
   - `esp32_mic_led_buzzer.ino` → **ESP32 DevKit**
   - `esp32_cam.ino` → **ESP32-CAM**
4. Jalankan backend FastAPI.
5. Kedua device akan otomatis mengirim data audio + frame ke server untuk dianalisis.

---

# 🏗️ Bagian dari Sistem Lebih Besar
Modul IoT ini digunakan dalam project:

**Argus — Smart Cheating Detection System**  
Menggabungkan:
- IoT (audio + video)
- FastAPI
- Streaming inference
- Dashboard Analytics

---

# 👩‍💻 Developer Notes
- Gunakan hotspot atau router yang sama untuk ESP32 & server.
- IP hotspot biasanya berubah–ubah, jadi pastikan cek ulang.
- Untuk ESP32-CAM: saat upload perlu tekan tombol **BOOT** (jika pakai FTDI/ESP32 DevKit sebagai programmer).
