#include <Arduino.h>
#include <driver/i2s.h>
#include "I2SMicSampler.h"
#include "config.h"
#include "CommandDetector.h"
#include "CommandProcessor.h"
#include <esp_task_wdt.h>
#include <SoftwareSerial.h>

SoftwareSerial mySerial(14, 12);

char buffer_[100];
int buffer_len;

#define DELAYFORWHITESPACE 400
#define WIFIPIN 4

// i2s config for reading from both channels of I2S
i2s_config_t i2sMemsConfigBothChannels = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = 16000,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_MIC_CHANNEL,
    .communication_format = i2s_comm_format_t(I2S_COMM_FORMAT_I2S),
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 64,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0};

// i2s microphone pins
i2s_pin_config_t i2s_mic_pins = {
    .bck_io_num = I2S_MIC_SERIAL_CLOCK,
    .ws_io_num = I2S_MIC_LEFT_RIGHT_CLOCK,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_MIC_SERIAL_DATA};

String convertToString(char *a, int size)
{
  int i;
  String s = "";
  for (i = 0; i < size; i++)
  {
    s = s + a[i];
  }
  return s;
}

// This task does all the heavy lifting for our application
void applicationTask(void *param)
{
  CommandDetector *commandDetector = static_cast<CommandDetector *>(param);

  const TickType_t xMaxBlockTime = pdMS_TO_TICKS(100);
  while (true)
  {
    buffer_len = 0;

    if (mySerial.available() >= 2)
    {
      for (int i = 0; mySerial.available(); i++)
      {
        buffer_[i] = mySerial.read();
        buffer_len++;
      }

      // int whiteSpaces = 1;
      // for (int i = 0; i < buffer_len; i++)
      // {
      //   if (buffer_[i] == ' ')
      //   {
      //     whiteSpaces++;
      //   }
      // }

      Serial.println(buffer_);
      Serial.println(convertToString(buffer_, buffer_len));
      Serial2.println(convertToString(buffer_, buffer_len));

      delay((buffer_len / 2) * DELAYFORWHITESPACE);

      ESP.restart();
    }

    // wait for some audio samples to arrive
    uint32_t ulNotificationValue = ulTaskNotifyTake(pdTRUE, xMaxBlockTime);
    if (ulNotificationValue > 0)
    {
      commandDetector->run();
    }
  }
}

void setup()
{
  pinMode(18, INPUT); // buzzer
  pinMode(WIFIPIN, OUTPUT);

  Serial.begin(9600);
  Serial2.begin(115200, SERIAL_8N1, RXp2, TXp2);

  mySerial.begin(115200);

  delay(1000);
  Serial.println("Starting up");

  // make sure we don't get killed for our long running tasks
  esp_task_wdt_init(10, false);
  I2SSampler *i2s_sampler = new I2SMicSampler(i2s_mic_pins, false);

  // create our application
  CommandDetector *commandDetector = new CommandDetector(i2s_sampler);

  // set up the i2s sample writer task
  TaskHandle_t applicationTaskHandle;
  xTaskCreatePinnedToCore(applicationTask, "Command Detect", 8192, commandDetector, 1, &applicationTaskHandle, 0);

  i2s_sampler->start(I2S_NUM_0, i2sMemsConfigBothChannels, applicationTaskHandle);
}

void loop()
{
  esp_task_wdt_init(30, false);
  vTaskDelay(pdMS_TO_TICKS(1000));
}