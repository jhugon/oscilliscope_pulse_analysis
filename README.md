# Oscilloscope Pulse Analysis

## Power supply setup for C-series 1 mm SiPM

### SiPM Power Supply Settings

- V: 27 V
- I: 5 mA
- V trip: 30 V
- I trip: 6 mA

### Amplifier Power Supply Settings

Using a separate channel for each of 2x [ZFL-500LN+ amplifiers](https://www.minicircuits.com/pdfs/ZFL-500LN+.pdf):

- V: 15 V
- I: 55 mA
- V trip: 17 V
- I trip: 60 mA

## Signal generator setup

A few quantized pulse sizes seem to be well populated for:

- Mode: pulse
- Frequency: 10 kHz
- HighL: 1.4 V
- LowL: 0 V
- Phase: 0
- Width: 16 ns

- Output: 50 ohm

with some red LED and series resistor (100 ohm?)
