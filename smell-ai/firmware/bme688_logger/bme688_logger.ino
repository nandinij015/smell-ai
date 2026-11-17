/*
  BME688 heater-cycle logger for ESP32  (smell-ai, README Part 3)

  Wiring (I2C):   BME688 3V3 -> ESP32 3V3
                  BME688 GND -> ESP32 GND
                  BME688 SDA -> ESP32 GPIO21
                  BME688 SCL -> ESP32 GPIO22

  What it does: steps the gas-sensor heater through 4 temperatures
  (200, 250, 300, 350 C), reads the gas resistance at each, and prints ONE
  line per cycle over USB serial (115200 baud):

     D,<ms>,<gas200>,<gas250>,<gas300>,<gas350>,<temp_C>,<humidity_%>,<pressure_hPa>

  Gas values are in ohms. Lines starting with '#' are info messages.
  Libraries (Arduino Library Manager): "Adafruit BME680 Library" + "Adafruit Unified Sensor".
  Board: "ESP32 Dev Module".
*/
#include <Wire.h>
#include <Adafruit_Sensor.h>
#include "Adafruit_BME680.h"

#define SDA_PIN 21
#define SCL_PIN 22

const uint16_t HEATER_TEMPS_C[4] = {200, 250, 300, 350};
const uint16_t HEATER_MS = 150;       // how long each heater step lasts
const uint32_t CYCLE_PAUSE_MS = 1000; // rest between cycles (also saves heater wear)

Adafruit_BME680 bme(&Wire);

// Look for devices on the I2C bus and say what we found.
void scanI2C() {
  Serial.println("# I2C scan:");
  int found = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      Serial.print("#   device at 0x");
      Serial.println(addr, HEX);
      found++;
    }
  }
  if (found == 0) {
    Serial.println("#   NOTHING found -> check wiring (SDA=21, SCL=22, 3V3, GND) and soldering");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1500);
  Serial.println("# bme688_logger starting");
  Wire.begin(SDA_PIN, SCL_PIN);
  scanI2C();

  // BME688 address is 0x76 or 0x77 depending on the breakout board
  bool ok = bme.begin(0x77) || bme.begin(0x76);
  if (!ok) {
    Serial.println("# ERROR: BME688 not found. Check the I2C scan above, wiring and solder joints.");
    while (true) delay(1000);
  }
  Serial.println("# BME688 found. Warm it up for 24 h before trusting readings (README step 2).");
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  Serial.println("# columns: D,ms,gas200,gas250,gas300,gas350,tempC,humidity,pressure_hPa");
}

void loop() {
  uint32_t gas[4];
  float tempC = 0, hum = 0, pres = 0;

  for (int i = 0; i < 4; i++) {
    bme.setGasHeater(HEATER_TEMPS_C[i], HEATER_MS);
    if (!bme.performReading()) {
      Serial.println("# WARNING: reading failed, skipping this cycle");
      delay(CYCLE_PAUSE_MS);
      return;
    }
    gas[i] = bme.gas_resistance;
    tempC = bme.temperature;
    hum = bme.humidity;
    pres = bme.pressure / 100.0;
  }

  Serial.print("D,");
  Serial.print(millis());
  for (int i = 0; i < 4; i++) {
    Serial.print(",");
    Serial.print(gas[i]);
  }
  Serial.print(",");
  Serial.print(tempC, 2);
  Serial.print(",");
  Serial.print(hum, 2);
  Serial.print(",");
  Serial.println(pres, 2);

  delay(CYCLE_PAUSE_MS);
}
