#!/usr/bin/env python3
"""
POX Controller with ML-based DDoS Detection

This script implements a POX controller with integrated machine learning models
for real-time DDoS attack detection in Software-Defined Networks (SDN).

The controller:
1. Collects traffic statistics from OpenFlow switches
2. Extracts features from the traffic data
3. Uses trained ML models to classify traffic as normal or attack
4. Installs flow rules to block malicious traffic

The Random Forest model is used as the primary classifier due to its high accuracy.
"""

import os
import sys
import time
import json
import threading
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# Simulated POX controller imports
# In a real implementation, these would be actual POX imports
class SimulatedPOX:
    """Simulated POX controller for demonstration purposes."""
    
    def __init__(self):
        self.switches = {}
        self.flows = {}
        self.blocked_flows = set()
        self.attack_count = 0
        self.normal_count = 0
        self.total_flows_processed = 0
        
        # Load ML model
        self.load_ml_model()
        
        print("POX Controller initialized with ML-based DDoS detection")
    
    def load_ml_model(self):
        """Load the trained machine learning model."""
        try:
            # Load the Random Forest model (best performer)
            self.model = joblib.load('models/random_forest.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            print("Successfully loaded Random Forest model")
        except FileNotFoundError:
            print("Error: Model files not found. Please run train_models.py first.")
            sys.exit(1)
    
    def add_switch(self, switch_id):
        """Add a switch to the controller."""
        self.switches[switch_id] = {
            'ports': {},
            'flows': {},
            'stats': {
                'packet_in_count': 0,
                'flow_mod_count': 0,
                'last_stats_time': time.time()
            }
        }
        print(f"Switch {switch_id} connected")
    
    def extract_features(self, flow_data):
        """
        Extract features from flow data for ML classification.
        
        Args:
            flow_data (dict): Raw flow data from the switch
            
        Returns:
            numpy.ndarray: Feature vector for ML model
        """
        # Extract features based on the 23 features in our dataset
        features = np.zeros(22)  # 22 features (excluding label)
        
        # Basic flow information
        features[0] = time.time()  # dt (decision time)
        features[1] = flow_data.get('switch_id', 0)  # switch
        features[2] = hash(flow_data.get('src_ip', '0.0.0.0')) % 1000  # src
        features[3] = hash(flow_data.get('dst_ip', '0.0.0.0')) % 1000  # dst
        features[4] = flow_data.get('packet_count', 0)  # pkt_count
        features[5] = flow_data.get('byte_count', 0)  # byte_count
        features[6] = flow_data.get('duration_sec', 0)  # duration
        features[7] = flow_data.get('duration_nsec', 0)  # duration_nsec
        
        # Calculate derived features
        total_duration = features[6] + features[7] / 1e9  # total_duration
        features[8] = total_duration
        features[9] = flow_data.get('flow_count', 1)  # flows
        
        # Avoid division by zero
        if total_duration > 0:
            features[10] = features[4] / total_duration  # pkt_rate
        else:
            features[10] = features[4]  # If duration is 0, use packet count as rate
        
        features[11] = flow_data.get('pair_flow', 0)  # pair_flow
        features[12] = flow_data.get('protocol', 0)  # protocol
        features[13] = flow_data.get('port', 0)  # port_no
        features[14] = flow_data.get('tx_bytes', 0)  # tx_bytes
        features[15] = flow_data.get('rx_bytes', 0)  # rx_bytes
        
        # Calculate bandwidth features
        if total_duration > 0:
            features[16] = features[14] * 8 / 1000 / total_duration  # tx_kbps
            features[17] = features[15] * 8 / 1000 / total_duration  # rx_kbps
            features[18] = features[16] + features[17]  # tot_kbps
        else:
            features[16] = 0
            features[17] = 0
            features[18] = 0
        
        features[19] = flow_data.get('packet_ins', 0)  # packetins
        
        # Calculate per-flow features
        if features[9] > 0:
            features[20] = features[5] / features[9]  # byte_per_flow
            features[21] = features[4] / features[9]  # packet_per_flow
        else:
            features[20] = features[5]
            features[21] = features[4]
        
        return features
    
    def classify_flow(self, flow_data):
        """
        Classify a flow as normal or attack using the ML model.
        
        Args:
            flow_data (dict): Flow data from the switch
            
        Returns:
            tuple: (is_attack, confidence)
        """
        # Extract features
        features = self.extract_features(flow_data)
        
        # Scale features
        scaled_features = self.scaler.transform(features.reshape(1, -1))
        
        # Predict using the model
        prediction = self.model.predict(scaled_features)[0]
        
        # Get prediction probability
        confidence = self.model.predict_proba(scaled_features)[0][prediction]
        
        is_attack = bool(prediction)
        
        return is_attack, confidence
    
    def handle_flow(self, flow_data):
        """
        Process a flow and take action based on classification.
        
        Args:
            flow_data (dict): Flow data from the switch
            
        Returns:
            bool: True if flow is blocked, False otherwise
        """
        # Classify the flow
        is_attack, confidence = self.classify_flow(flow_data)
        
        # Generate flow ID
        flow_id = f"{flow_data.get('src_ip')}:{flow_data.get('src_port')}->{flow_data.get('dst_ip')}:{flow_data.get('dst_port')}"
        
        # Update counters
        self.total_flows_processed += 1
        
        if is_attack:
            self.attack_count += 1
            
            # Block the flow if confidence is high enough
            if confidence > 0.8:
                self.blocked_flows.add(flow_id)
                
                # Log the attack
                attack_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'flow_id': flow_id,
                    'src_ip': flow_data.get('src_ip'),
                    'dst_ip': flow_data.get('dst_ip'),
                    'confidence': confidence,
                    'features': {
                        'pkt_rate': flow_data.get('packet_count', 0) / max(flow_data.get('duration_sec', 0.1), 0.1),
                        'byte_count': flow_data.get('byte_count', 0),
                        'flow_count': flow_data.get('flow_count', 1)
                    }
                }
                
                print(f"ATTACK DETECTED: {flow_id} (Confidence: {confidence:.2f})")
                print(f"Blocking malicious flow: {flow_id}")
                
                # In a real implementation, this would install a flow rule to block the traffic
                return True
        else:
            self.normal_count += 1
        
        return False
    
    def print_stats(self):
        """Print controller statistics."""
        total = self.normal_count + self.attack_count
        if total > 0:
            attack_percentage = (self.attack_count / total) * 100
        else:
            attack_percentage = 0
            
        print("\n--- Controller Statistics ---")
        print(f"Total flows processed: {self.total_flows_processed}")
        print(f"Normal flows: {self.normal_count}")
        print(f"Attack flows: {self.attack_count} ({attack_percentage:.2f}%)")
        print(f"Blocked flows: {len(self.blocked_flows)}")
        print("-----------------------------\n")
    
    def start(self):
        """Start the controller and simulate traffic."""
        print("Starting POX controller with ML-based DDoS detection...")
        
        # Add some simulated switches
        for i in range(1, 6):
            self.add_switch(i)
        
        # Start stats printing thread
        stats_thread = threading.Thread(target=self._stats_thread)
        stats_thread.daemon = True
        stats_thread.start()
        
        # Simulate traffic
        self._simulate_traffic()
    
    def _stats_thread(self):
        """Thread to periodically print statistics."""
        while True:
            time.sleep(5)
            self.print_stats()
    
    def _simulate_traffic(self):
        """Simulate network traffic for demonstration."""
        print("Simulating network traffic...")
        
        # Create some sample flows
        normal_flows = [
            {'switch_id': 1, 'src_ip': '192.168.1.10', 'src_port': 12345, 'dst_ip': '10.0.0.1', 'dst_port': 80, 
             'protocol': 6, 'packet_count': 10, 'byte_count': 1500, 'duration_sec': 2, 'duration_nsec': 0,
             'flow_count': 1, 'tx_bytes': 1000, 'rx_bytes': 500, 'packet_ins': 5, 'port': 1},
            {'switch_id': 2, 'src_ip': '192.168.1.11', 'src_port': 23456, 'dst_ip': '10.0.0.2', 'dst_port': 443, 
             'protocol': 6, 'packet_count': 15, 'byte_count': 2200, 'duration_sec': 3, 'duration_nsec': 0,
             'flow_count': 1, 'tx_bytes': 1200, 'rx_bytes': 1000, 'packet_ins': 7, 'port': 2}
        ]
        
        attack_flows = [
            {'switch_id': 3, 'src_ip': '172.16.0.1', 'src_port': 34567, 'dst_ip': '10.0.0.3', 'dst_port': 80, 
             'protocol': 6, 'packet_count': 1000, 'byte_count': 60000, 'duration_sec': 1, 'duration_nsec': 0,
             'flow_count': 50, 'tx_bytes': 58000, 'rx_bytes': 2000, 'packet_ins': 500, 'port': 1},
            {'switch_id': 4, 'src_ip': '172.16.0.2', 'src_port': 45678, 'dst_ip': '10.0.0.4', 'dst_port': 53, 
             'protocol': 17, 'packet_count': 800, 'byte_count': 48000, 'duration_sec': 1, 'duration_nsec': 0,
             'flow_count': 40, 'tx_bytes': 46000, 'rx_bytes': 2000, 'packet_ins': 400, 'port': 2}
        ]
        
        try:
            while True:
                # Process some normal flows
                for flow in normal_flows:
                    # Add some randomness
                    flow['packet_count'] += np.random.randint(1, 10)
                    flow['byte_count'] += np.random.randint(100, 1000)
                    flow['duration_sec'] += 1
                    self.handle_flow(flow)
                    time.sleep(0.5)
                
                # Process some attack flows
                for flow in attack_flows:
                    # Add some randomness
                    flow['packet_count'] += np.random.randint(100, 500)
                    flow['byte_count'] += np.random.randint(5000, 20000)
                    flow['duration_sec'] += 1
                    self.handle_flow(flow)
                    time.sleep(0.5)
        
        except KeyboardInterrupt:
            print("\nController stopped by user")

def main():
    """Main function to start the POX controller."""
    # Check if models exist
    if not os.path.exists('models/random_forest.pkl'):
        print("Error: ML models not found. Please run train_models.py first.")
        print("You can run: python src/ml/train_models.py")
        return
    
    # Create controller instance
    controller = SimulatedPOX()
    
    # Start the controller
    controller.start()

if __name__ == "__main__":
    main()