import time
import requests
import asyncio
from collections import deque
from loguru import logger
from collector.icmp_prober import ICMPProber
try:
    from collector.packet_sniffer import PacketSniffer
except ImportError:
    PacketSniffer = None

def collect_loop():
    logger.info("Starting REALTIME network collector loop...")
    url = "http://api:8000/predict"
    # We need 350 data points of history to run the FeaturePipeline
    history = deque(maxlen=350)
    
    prober = ICMPProber(timeout=1.0)
    target = "8.8.8.8" # Google DNS for consistent baseline measuring
    
    # Instantiate packet sniffer if available
    sniffer = None
    packet_queue = None
    # We create a new event loop just for the asyncio queue to prevent loop closure bugs
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    if PacketSniffer:
        packet_queue = asyncio.Queue()
        sniffer = PacketSniffer(queue=packet_queue)
        try:
            sniffer.start()
        except Exception as e:
            logger.warning("Could not start packet sniffer: {}", e)
            sniffer = None
    
    # Pre-populate history
    now = time.time() - 350
    for i in range(350):
        now += 1
        history.append({
            "timestamp": now,
            "rtt_ms": 15.0 + (i % 5),
            "jitter_ms": 1.0,
            "packet_loss": 0.0,
            "bytes_per_sec": 5000,
            "packets_per_sec": 100
        })
        
    while True:
        try:
            start_time = time.time()
            
            # --- 1. Ping / RTT ---
            result = prober.probe(target)
            rtt = result.get("rtt_ms")
            packet_loss = 1.0 if rtt is None else 0.0
            if rtt is None:
                rtt = 100.0 # arbitrary high value for timeout penalty
                
            last_rtt = history[-1]['rtt_ms']
            jitter = abs(rtt - last_rtt)
            
            # --- 2. Bandwidth Sniffing ---
            bytes_sent = 5000
            packets_sent = 100
            
            if sniffer and packet_queue:
                q_size = packet_queue.qsize()
                if q_size > 0:
                    bytes_sent = 0
                    packets_sent = 0
                    for _ in range(q_size):
                        pkt = packet_queue.get_nowait()
                        bytes_sent += pkt.get("length", 0)
                        packets_sent += 1
            
            history.append({
                "timestamp": start_time,
                "rtt_ms": rtt,
                "jitter_ms": jitter,
                "packet_loss": packet_loss,
                "bytes_per_sec": bytes_sent,
                "packets_per_sec": packets_sent
            })
            
            payload = list(history)
            r = requests.post(url, json=payload, timeout=2)
            logger.info("Live Predict -> {} | Actual RTT: {}ms | Network Bps: {}", r.status_code, rtt, bytes_sent)
            
        except requests.exceptions.ConnectionError:
            logger.warning("API not reachable, retrying...")
        except Exception as e:
            logger.error("Error: {}", e)
        
        # Try to enforce 1 second polling interval cleanly
        elapsed = time.time() - start_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)

if __name__ == "__main__":
    collect_loop()
