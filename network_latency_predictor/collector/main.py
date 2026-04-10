import time
import requests
from collections import deque
from loguru import logger
import struct
import socket

# Quick ICMP generator for simulation of real packets natively
def collect_live():
    # Because making raw socket ICMP request requires sudo/root locally on Mac but works in Docker
    # We will simulate the live capture by mimicking the format precisely for the API
    pass

if __name__ == "__main__":
    logger.info("Starting network collector loop...")
    
    url = "http://api:8000/predict"
    # We need 350 data points of history to run the FeaturePipeline
    history = deque(maxlen=350)
    
    # Pre-populate history
    now = time.time() - 350
    for i in range(350):
        now += 1
        history.append({
            "timestamp": now,
            "rtt_ms": 15.0 + (i % 5),
            "jitter_ms": 1.0,
            "packet_loss": 0.0,
            "bytes_per_sec": 5000 + i,
            "packets_per_sec": 100
        })
        
    while True:
        try:
            now = time.time()
            history.append({
                "timestamp": now,
                "rtt_ms": 15.0 + (int(now) % 10),
                "jitter_ms": 1.2,
                "packet_loss": 0.0,
                "bytes_per_sec": 5500,
                "packets_per_sec": 120
            })
            
            payload = list(history)
            r = requests.post(url, json=payload, timeout=2)
            logger.info("API Predict Response: {} -> {}ms (10s)", r.status_code, r.json().get('rtt_10s'))
        except requests.exceptions.ConnectionError:
            logger.warning("API not reachable, retrying in 2s...")
        except Exception as e:
            logger.error("Error: {}", e)
        
        time.sleep(2)
