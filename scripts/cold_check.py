"""Nemotron's E6 sweep ran with -c 1048576 after hours of load and read 12% slower than the
cold 234k fill in E2. Two candidate causes: the 1M allocation, or thermal throttling. This
repeats the ladder at -c 262144 from cold cards, with GPU telemetry, to separate them."""
import wave10 as w
w.wait_cool(45, 1800)
w.LEVELS = [4096, 16384, 32768, 65536, 131072]
w.window_sweep("nemotron-256k-cold", w.NEMO, [262144])
