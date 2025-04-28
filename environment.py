import torch
import math
import pandas as pd

SPEED_OF_LIGHT = 3e8  # m/s

class ResourceEnvironment:
    def __init__(self, bs_state, ue_csv_path):
        self.bs_state = bs_state.to(torch.float32)
        self.ue_state = self.load_ue_data(ue_csv_path).to(torch.float32)
        self.allocations = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_bs_properties()

    def load_ue_data(self, csv_path):
        df = pd.read_csv(csv_path)
        return torch.tensor(df[['x', 'y', 'vx', 'vy', 'bandwidth', 'latency', 'slice_type']].values)

    def update_bs_properties(self, max_power=None, frequency=2.4e9, tx_antenna_gain=1, rx_antenna_gain=1):
        if max_power is not None:
            self.bs_state[:, 3] = torch.tensor(max_power, device=self.device, dtype=torch.float32)

        sensitivity = 1e-10
        wavelength = SPEED_OF_LIGHT / frequency
        for i in range(self.bs_state.shape[0]):
            max_power = self.bs_state[i, 3].item()
            max_dist = (wavelength / (4 * math.pi)) * math.sqrt(
                (max_power * tx_antenna_gain * rx_antenna_gain) / sensitivity)
            self.bs_state[i, 4] = max_dist

    def calculate_distance_and_snr(self):
        bs_x, bs_y = self.bs_state[:, 0], self.bs_state[:, 1]
        ue_positions = self.ue_state[:, :2]

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

    def apply_allocations(self, allocations):
        self.allocations = allocations

    def compute_loss(self):
        if self.allocations is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0, 0.0

        bs_assignment, bandwidth_allocation = self.allocations
        N_ue = self.ue_state.shape[0]
        N_bs = self.bs_state.shape[0]

        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        bs_bandwidth_consumed = torch.zeros(N_bs, device=self.device)
        bs_power_consumed = torch.zeros(N_bs, device=self.device)
        outage_count = 0
        bandwidth_deficit = torch.tensor(0.0, device=self.device)
        total_allocated = torch.tensor(0.0, device=self.device)

        # Loss weights
        weights = {
            'outage': 10.0,
            'bw_deficit': 100.0,
            'bw_exceed': 100.0,
            'power_exceed': 10.0,
            'good_assignment': -5.0,
            'slice': 15.0,
            'distance': 1000.0,
            'snr': 200.0,
            'load_balance': 10.0
        }

        # Calculate SNR and distances for all UE-BS pairs
        distances, snr_db = self.calculate_distance_and_snr()

        for i in range(N_ue):
            ue_need = self.ue_state[i, 4]
            assigned_bs = bs_assignment[i]
            assigned_bw = bandwidth_allocation[i]

            # Never allocate more than needed
            assigned_bw = torch.min(assigned_bw, ue_need)

            if assigned_bs == -1:
                loss = loss + weights['outage']
                outage_count += 1
                bandwidth_deficit = bandwidth_deficit + ue_need
                continue

            # Bandwidth calculations
            bw_diff = torch.relu(ue_need - assigned_bw)
            bandwidth_deficit = bandwidth_deficit + bw_diff
            total_allocated += assigned_bw
            loss = loss + weights['bw_deficit'] * (bw_diff / (ue_need + 1e-6))

            # Distance calculations
            distance = distances[i, assigned_bs]
            distance_ratio = distance / self.bs_state[assigned_bs, 4]
            loss = loss + weights['distance'] * torch.relu(distance_ratio - 0.8) * 5

            # SNR quality
            current_snr = snr_db[i, assigned_bs]
            min_snr = 10.0  # Minimum acceptable SNR in dB
            snr_penalty = torch.relu(min_snr - current_snr)
            loss = loss + weights['snr'] * snr_penalty

            # Track resources
            bs_bandwidth_consumed[assigned_bs] += assigned_bw
            bs_power_consumed[assigned_bs] += self.calculate_required_power(distance, assigned_bw)

        # BS constraints
        for j in range(N_bs):
            available_bw = self.bs_state[j, 2]
            if bs_bandwidth_consumed[j] > available_bw:
                exceed = (bs_bandwidth_consumed[j] - available_bw) / (available_bw + 1e-6)
                loss = loss + weights['bw_exceed'] * exceed ** 2

            max_power = self.bs_state[j, 3]
            if bs_power_consumed[j] > max_power:
                exceed = (bs_power_consumed[j] - max_power) / (max_power + 1e-6)
                loss = loss + weights['power_exceed'] * exceed ** 2

        # Load balancing penalty
        mean_load = torch.mean(bs_bandwidth_consumed / self.bs_state[:, 2])
        load_imbalance = torch.std(bs_bandwidth_consumed / self.bs_state[:, 2])
        loss = loss + weights['load_balance'] * load_imbalance

        total_bw_need = torch.sum(self.ue_state[:, 4])
        if total_bw_need > 0:
            loss = loss + weights['bw_deficit'] * (bandwidth_deficit / total_bw_need)

        return loss, outage_count, bandwidth_deficit.item()

    def calculate_required_power(self, distance, bandwidth):
        path_loss_exponent = 3.0
        reference_distance = 1.0
        reference_loss_db = 20 * math.log10(4 * math.pi * reference_distance * 2.4e9 / SPEED_OF_LIGHT)
        reference_loss = 10 ** (reference_loss_db / 10)

        if distance <= reference_distance:
            path_loss = reference_loss
        else:
            path_loss = reference_loss * (distance / reference_distance) ** path_loss_exponent

        return (1e-10 * path_loss) * (bandwidth / 1e6)