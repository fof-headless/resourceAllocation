import random
import pandas as pd
import numpy as np


def generate_synthetic_ue_data(num_ues, num_timesteps):
    # Define slice type ranges
    slice_types = [0, 1, 2]

    # Initialize lists to store data
    all_ues_data = []

    for _ in range(num_ues):
        # Randomly pick slice type
        slice_type = random.choice(slice_types)

        # Initial position and velocity
        x, y = np.random.uniform(-5000, 5000), np.random.uniform(-5000, 5000)
        vx, vy = np.random.uniform(-5, 5), np.random.uniform(-5, 5)

        # Generate initial bandwidth and latency based on slice type
        if slice_type == 0:
            bandwidth = random.randint(50, 100)  # 50 - 100 Mbps
            latency = random.randint(10, 50)  # 10 - 50 ms
        elif slice_type == 1:
            bandwidth = random.randint(1, 5)  # 1 - 5 Mbps
            latency = random.randint(50, 100)  # 50 - 100 ms
        elif slice_type == 2:
            bandwidth = random.randint(5, 20)  # 5 - 20 Mbps
            latency = random.randint(1, 10)  # 1 - 10 ms

        # Store the initial data for time step 0
        ues_data = [[x, y, vx, vy, bandwidth, latency, slice_type, 0]]

        # Simulate movement and network conditions over time
        for t in range(1, num_timesteps):
            # Update position based on velocity
            x += vx
            y += vy

            # Introduce some randomness in velocity (simulate random movement changes)
            vx += np.random.uniform(-0.5, 0.5)  # Small random change in velocity
            vy += np.random.uniform(-0.5, 0.5)

            # Simulate temporal dynamics for bandwidth and latency
            if slice_type == 0:
                bandwidth = random.randint(50, 100)
                latency = random.randint(10, 50)
            elif slice_type == 1:
                bandwidth = random.randint(1, 5)
                latency = random.randint(50, 100)
            elif slice_type == 2:
                bandwidth = random.randint(5, 20)
                latency = random.randint(1, 10)

            # Add the data for the current time step
            ues_data.append([x, y, vx, vy, bandwidth, latency, slice_type, t])

        # Append the data for this UE to the overall list
        all_ues_data.extend(ues_data)

    # Create a DataFrame from the generated data
    df = pd.DataFrame(all_ues_data, columns=['x', 'y', 'vx', 'vy', 'bandwidth', 'latency', 'slice_type', 'time'])
    return df


# Example: Generate data for two different time instances with temporal dynamics
num_ues = 150
num_timesteps = 1  # Number of time steps for each UE

# Generate synthetic data with temporal dynamics for each time instance
ue_data_t1 = generate_synthetic_ue_data(num_ues, num_timesteps)
ue_data_t2 = generate_synthetic_ue_data(num_ues, num_timesteps)

# Save the synthetic data to CSV files for each time instance
ue_data_t1.to_csv('ue_data_t1.csv', index=False)
ue_data_t2.to_csv('ue_data_t2.csv', index=False)

ue_data_t1 = pd.read_csv('ue_data_t1.csv')

# Assume time step is 2 minutes (120 seconds)
time_step = 120  # 2 minutes

# Calculate the final positions after 2 minutes
# New position = Initial position + (velocity * time)
ue_data_t2 = ue_data_t1.copy()  # Start with the same data

# Update the positions (x, y) based on speed (vx, vy) over 2 minutes
ue_data_t2['x'] += ue_data_t2['vx'] * time_step
ue_data_t2['y'] += ue_data_t2['vy'] * time_step

# Save the modified data to a final CSV (ue_data_t2.csv)
ue_data_t2.to_csv('ue_data_t2.csv', index=False)

# Show a preview of the updated data
print("Updated data for Time Instance 2 (t2):")
print(ue_data_t1.head())
print("Updated data for Time Instance 2 (t2):")
print(ue_data_t2.head())