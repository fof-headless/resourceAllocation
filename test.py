import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from allocator import AllocatorNN
from environment import ResourceEnvironment
import math
import time

from final.environment import SPEED_OF_LIGHT

SPEED_OF_LIGHT = 3e8
class Predictor:
    def __init__(self, model_path, bs_positions, bs_bandwidths, bs_powers):
        """
        Initialize the predictor with trained model and base station configuration

        Args:
            model_path: Path to the trained model weights
            bs_positions: List of (x,y) tuples for base station positions
            bs_bandwidths: List of bandwidth capacities for each BS (in Mbps)
            bs_powers: List of max transmission powers for each BS (in Watts)
        """
        self.device = torch.device("mps")

        # Initialize base station state
        bs_list = []
        for (x, y), bw, power in zip(bs_positions, bs_bandwidths, bs_powers):
            bs_list.append([x, y, bw, power, 0.0])  # Last 0.0 will be filled with max distance

        self.bs_state = torch.tensor(bs_list, dtype=torch.float32).to(self.device)
        self.num_bs = len(bs_positions)

        # Initialize model
        self.model = AllocatorNN(
            ue_input_dim=7,
            bs_input_dim=5,
            hidden_dim=128,
            num_bs=self.num_bs
        ).to(self.device)

        # Load trained weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        start_time = time.time()
        self.model.eval()
        end_time = time.time()
        # Update BS properties (calculates max distances)
        elapsed_time_ms = (end_time - start_time) * 1000

        print(f"x() took {elapsed_time_ms:.2f} milliseconds to execute.")
        self.update_bs_properties(bs_powers)

    def update_bs_properties(self, max_power=None, frequency=2.4e9, tx_antenna_gain=1, rx_antenna_gain=1):
        """Update base station transmission properties"""
        SPEED_OF_LIGHT = 3e8  # m/s

        if max_power is not None:
            if isinstance(max_power, (list, tuple)):
                for i, power in enumerate(max_power):
                    self.bs_state[i, 3] = torch.tensor(power, device=self.device, dtype=torch.float32)
            else:
                self.bs_state[:, 3] = torch.tensor(max_power, device=self.device, dtype=torch.float32)

        sensitivity = 1e-10
        wavelength = SPEED_OF_LIGHT / frequency

        for i in range(self.bs_state.shape[0]):
            max_power = self.bs_state[i, 3].item()
            max_dist = (wavelength / (4 * math.pi)) * math.sqrt(
                (max_power * tx_antenna_gain * rx_antenna_gain) / sensitivity)
            self.bs_state[i, 4] = max_dist

    def prepare_ue_data(self, ue_data):
        """
        Prepare UE data from dictionary or DataFrame format

        Expected UE data format:
        Each UE should have:
        - x, y: position coordinates
        - vx, vy: velocity components (can be 0 if not moving)
        - bandwidth: requested bandwidth in Mbps
        - latency: maximum acceptable latency in ms
        - slice_type: network slice type (0=best effort, 1=low latency, 2=high throughput)

        Returns:
            torch.Tensor of shape (N_ue, 7)
        """
        if isinstance(ue_data, pd.DataFrame):
            # Convert from DataFrame
            return torch.tensor(ue_data[['x', 'y', 'vx', 'vy', 'bandwidth', 'latency', 'slice_type']].values,
                                dtype=torch.float32).to(self.device)
        elif isinstance(ue_data, dict):
            # Convert from dictionary of lists
            return torch.tensor([
                ue_data['x'], ue_data['y'], ue_data['vx'], ue_data['vy'],
                ue_data['bandwidth'], ue_data['latency'], ue_data['slice_type']
            ], dtype=torch.float32).T.to(self.device)
        elif isinstance(ue_data, (list, np.ndarray)):
            # Direct numpy array or list input
            return torch.tensor(ue_data, dtype=torch.float32).to(self.device)
        else:
            raise ValueError("UE data must be pandas DataFrame, dictionary, or array-like")

    def calculate_distance_and_snr(self, ue_state):
        """Calculate distances and SNR between UEs and BSs"""
        bs_x, bs_y = self.bs_state[:, 0], self.bs_state[:, 1]
        ue_positions = ue_state[:, :2]

        distances = torch.sqrt(
            (ue_positions[:, 0].unsqueeze(1) - bs_x.unsqueeze(0)) ** 2 +
            (ue_positions[:, 1].unsqueeze(1) - bs_y.unsqueeze(0)) ** 2
        )

        # Calculate SNR for each UE-BS pair
        tx_power = self.bs_state[:, 3].unsqueeze(0)  # Shape: (1, N_bs)
        noise_power = 1e-10  # Fixed noise power
        path_loss = (4 * math.pi * distances * 2.4e9 / SPEED_OF_LIGHT) ** 2
        snr = tx_power / (path_loss * noise_power)
        snr_db = 10 * torch.log10(snr)

        return distances, snr_db

    def predict(self, ue_data):
        """
        Make predictions for UE allocations

        Args:
            ue_data: Input UE data in DataFrame, dictionary, or array format

        Returns:
            tuple: (bs_assignments, bandwidth_allocations)
            - bs_assignments: List of assigned BS indices (-1 means no assignment)
            - bandwidth_allocations: List of allocated bandwidths in Mbps
        """
        with torch.no_grad():
            ue_state = self.prepare_ue_data(ue_data)
            distances, snr_db = self.calculate_distance_and_snr(ue_state)

            bs_assignment, bw_alloc = self.model(ue_state, self.bs_state, distances, snr_db)

            return bs_assignment.cpu().numpy(), bw_alloc.cpu().numpy()

    def visualize_allocation(self, ue_data, bs_assignment, bandwidth_allocation):
        """
        Visualize the allocation results

        Args:
            ue_data: Original UE input data
            bs_assignment: Array of BS assignments
            bandwidth_allocation: Array of bandwidth allocations
        """
        ue_state = self.prepare_ue_data(ue_data).cpu().numpy()

        plt.figure(figsize=(12, 10))

        # Plot Base Stations
        bs_x = self.bs_state[:, 0].cpu().numpy()
        bs_y = self.bs_state[:, 1].cpu().numpy()
        plt.scatter(bs_x, bs_y, c='black', s=300, marker='^', label='Base Stations')

        # Plot UEs
        ue_x = ue_state[:, 0]
        ue_y = ue_state[:, 1]

        # Calculate allocation ratios for coloring
        allocation_ratios = []
        for i in range(ue_state.shape[0]):
            if bs_assignment[i] != -1:
                ratio = bandwidth_allocation[i] / ue_state[i, 4]
                allocation_ratios.append(min(ratio, 1.0))
            else:
                allocation_ratios.append(0)

        # Create a colormap: red (0) -> yellow (0.5) -> green (1)
        colors = plt.cm.rainbow([x / 2 + 0.5 for x in allocation_ratios])

        # Plot connections
        for i in range(ue_state.shape[0]):
            if bs_assignment[i] != -1:
                bs_idx = bs_assignment[i]
                plt.plot([ue_x[i], bs_x[bs_idx]], [ue_y[i], bs_y[bs_idx]],
                         color=colors[i], alpha=0.4, linewidth=1)

        # Plot UEs with color indicating allocation ratio
        sc = plt.scatter(ue_x, ue_y, c=allocation_ratios, cmap='rainbow',
                         vmin=0, vmax=1, s=100, label='UEs')
        plt.colorbar(sc, label='Bandwidth Allocation Ratio (Actual/Requested)')

        # Add coverage circles
        for i, (x, y) in enumerate(zip(bs_x, bs_y)):
            max_dist = self.bs_state[i, 4].item()
            circle = patches.Circle((x, y), max_dist,
                                    fill=False, color='blue', alpha=0.2, linestyle='--')
            plt.gca().add_patch(circle)

            # Annotate BS with its ID
            plt.text(x, y, f"BS {i}", ha='center', va='center',
                     fontsize=10, weight='bold', color='white')

        plt.title("Network Allocation Prediction\n"
                  f"Colors show bandwidth fulfillment (red=outage, green=fully allocated)")
        plt.xlabel("X Coordinate (meters)")
        plt.ylabel("Y Coordinate (meters)")
        plt.grid(True)
        plt.axis('equal')
        plt.tight_layout()
        plt.show()

    def generate_report(self, ue_data, bs_assignment, bandwidth_allocation):
        """
        Generate a detailed allocation report

        Args:
            ue_data: Original UE input data
            bs_assignment: Array of BS assignments
            bandwidth_allocation: Array of bandwidth allocations
        """
        if isinstance(ue_data, pd.DataFrame):
            ue_state = ue_data[['x', 'y', 'vx', 'vy', 'bandwidth', 'latency', 'slice_type']].values
        else:
            ue_state = self.prepare_ue_data(ue_data).cpu().numpy()

        # Calculate bandwidth usage per BS
        bs_bw_used = np.zeros(self.num_bs)
        bs_ue_counts = np.zeros(self.num_bs, dtype=int)

        # Prepare UE report data
        ue_reports = []
        for i in range(ue_state.shape[0]):
            ue_need = ue_state[i, 4]
            assigned_bs = bs_assignment[i]
            assigned_bw = bandwidth_allocation[i] if assigned_bs != -1 else 0

            if assigned_bs != -1:
                bs_bw_used[assigned_bs] += assigned_bw
                bs_ue_counts[assigned_bs] += 1

            ue_reports.append({
                'id': i,
                'pos': (ue_state[i, 0], ue_state[i, 1]),
                'need': ue_need,
                'assigned_bw': assigned_bw,
                'bs': assigned_bs,
                'deficit': max(0, ue_need - assigned_bw)
            })

        # Print BS bandwidth reports
        print("\n=== Base Station Bandwidth Utilization ===")
        for bs_idx in range(self.num_bs):
            total_bw = self.bs_state[bs_idx, 2].item()
            used_bw = bs_bw_used[bs_idx]
            util_percent = (used_bw / total_bw) * 100 if total_bw > 0 else 0

            print(f"BS {bs_idx} [Position: ({self.bs_state[bs_idx, 0]:.1f}, {self.bs_state[bs_idx, 1]:.1f})]")
            print(f"  UEs Connected: {bs_ue_counts[bs_idx]}")
            print(f"  Bandwidth: {used_bw:.1f}/{total_bw:.1f} Mbps ({util_percent:.1f}% utilized)")
            print(f"  Available: {total_bw - used_bw:.1f} Mbps remaining")

        # Print UE allocation details
        print("\n=== Detailed UE Bandwidth Allocation ===")
        print(
            f"{'UE ID':<6} | {'Position':<15} | {'BS':<4} | {'Requested':<10} | {'Allocated':<10} | {'Deficit':<10} | {'Status':<12}")
        print("-" * 80)

        for ue in sorted(ue_reports, key=lambda x: x['bs'] if x['bs'] != -1 else float('inf')):
            status = "OUTAGE" if ue['bs'] == -1 else \
                "FULL" if ue['deficit'] == 0 else \
                    "PARTIAL"

            print(f"{ue['id']:<6} | ({ue['pos'][0]:>5.1f}, {ue['pos'][1]:>5.1f}) | "
                  f"{ue['bs'] if ue['bs'] != -1 else 'N/A':<4} | "
                  f"{ue['need']:>9.1f} Mbps | "
                  f"{ue['assigned_bw']:>9.1f} Mbps | "
                  f"{ue['deficit']:>9.1f} Mbps | "
                  f"{status:<12}")

        # Calculate overall metrics
        total_need = sum(ue['need'] for ue in ue_reports)
        total_allocated = sum(ue['assigned_bw'] for ue in ue_reports)
        outage_count = sum(1 for ue in ue_reports if ue['bs'] == -1)

        print("\n=== System-wide Metrics ===")
        print(f"Total UEs: {len(ue_reports)}")
        print(f"Outage UEs: {outage_count} ({outage_count / len(ue_reports) * 100:.1f}%)")
        print(f"Bandwidth Demand: {total_need:.1f} Mbps")
        print(f"Bandwidth Allocated: {total_allocated:.1f} Mbps ({total_allocated / total_need * 100:.1f}% of demand)")
        print(f"Total Deficit: {total_need - total_allocated:.1f} Mbps")


# Example usage
if __name__ == "__main__":
    # 1. Initialize predictor with your BS configuration
    predictor = Predictor(
        model_path="besttt.pt",  # Path to your trained model
        bs_positions=[(0.0, 3000.0), (0.0, -3000.0)],  # Same as training
        bs_bandwidths=[900, 900],  # Mbps
        bs_powers=[10, 10]  # Watts
    )

    # 2. Load UE data from CSV file
    # Assuming your CSV file has columns: 'x', 'y', 'vx', 'vy', 'bandwidth', 'latency', 'slice_type'
    ue_data = pd.read_csv("ue_data_t1.csv")  # Replace "ue_data.csv" with your CSV file path
    start_time = time.time()
    # 3. Make predictions
    bs_assignments, bw_allocations = predictor.predict(ue_data)
    end_time = time.time()
    # 4. Visualize results
    predictor.visualize_allocation(ue_data, bs_assignments, bw_allocations)

    # 5. Generate detailed report
    predictor.generate_report(ue_data, bs_assignments, bw_allocations)
    elapsed_time_ms = (end_time - start_time) * 1000
    print(f"x() took {elapsed_time_ms:.2f} milliseconds to execute.")