import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from environment import ResourceEnvironment
from allocator import AllocatorNN
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def train(model, env, epochs=1000, lr=1e-3, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience // 2, factor=0.5)

    best_loss = float('inf')
    no_improve = 0
    loss_history = []
    outage_history = []
    deficit_history = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        distances, snr_db = env.calculate_distance_and_snr()
        bs_assignment, bw_alloc = model(env.ue_state, env.bs_state, distances, snr_db)

        env.apply_allocations((bs_assignment, bw_alloc))
        loss, outage_count, bw_deficit = env.compute_loss()

        # Record history for plotting
        loss_history.append(loss.item())
        outage_history.append(outage_count / env.ue_state.shape[0] * 100)
        deficit_history.append(bw_deficit / torch.sum(env.ue_state[:, 4]).item() * 100)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step(loss)

        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            no_improve = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0:
            total_bw = torch.sum(env.ue_state[:, 4]).item()
            print(f"Epoch {epoch} Loss: {loss.item():.4f} "
                  f"Outage: {(outage_count / env.ue_state.shape[0]) * 100:.2f}% "
                  f"BW Deficit: {(bw_deficit / total_bw) * 100 if total_bw > 0 else 0:.2f}%")

    # Plot training history
    plot_training_history(loss_history, outage_history, deficit_history)

    # Load best model and evaluate
    model.load_state_dict(torch.load('best_model.pt'))
    evaluate_model(model, env)


def evaluate_model(model, env):
    model.eval()
    with torch.no_grad():
        distances, snr_db = env.calculate_distance_and_snr()
        bs_assignment, bandwidth_allocation = model(env.ue_state, env.bs_state, distances, snr_db)

        # Detailed reporting
        print_detailed_stats(env, bs_assignment.cpu(), bandwidth_allocation.cpu())
        visualize_network(env, bs_assignment.cpu(), bandwidth_allocation.cpu())

        env.apply_allocations((bs_assignment, bandwidth_allocation))
        _, outage, deficit = env.compute_loss()

    print(f"\nFinal Results:")
    print(f"Outage: {(outage / env.ue_state.shape[0]) * 100:.2f}%")
    print(f"Bandwidth Deficit: {(deficit / torch.sum(env.ue_state[:, 4]).item()) * 100:.2f}%")


def plot_training_history(loss_history, outage_history, deficit_history):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 3, 2)
    plt.plot(outage_history)
    plt.title('Outage Percentage')
    plt.xlabel('Epoch')
    plt.ylabel('% UEs in Outage')

    plt.subplot(1, 3, 3)
    plt.plot(deficit_history)
    plt.title('Bandwidth Deficit')
    plt.xlabel('Epoch')
    plt.ylabel('% Deficit')

    plt.tight_layout()
    plt.show()


def print_detailed_stats(env, bs_assignment, bandwidth_allocation):
    # Calculate bandwidth usage per BS
    bs_bw_used = torch.zeros(env.bs_state.shape[0])
    bs_ue_counts = torch.zeros(env.bs_state.shape[0], dtype=torch.int32)

    # Prepare UE report data
    ue_reports = []
    for i in range(env.ue_state.shape[0]):
        ue_need = env.ue_state[i, 4].item()
        assigned_bs = bs_assignment[i].item()
        assigned_bw = bandwidth_allocation[i].item() if assigned_bs != -1 else 0

        if assigned_bs != -1:
            bs_bw_used[assigned_bs] += assigned_bw
            bs_ue_counts[assigned_bs] += 1

        ue_reports.append({
            'id': i,
            'pos': (env.ue_state[i, 0].item(), env.ue_state[i, 1].item()),
            'need': ue_need,
            'assigned_bw': assigned_bw,
            'bs': assigned_bs,
            'deficit': max(0, ue_need - assigned_bw)
        })

    # Print BS bandwidth reports
    print("\n=== Base Station Bandwidth Utilization ===")
    for bs_idx in range(env.bs_state.shape[0]):
        total_bw = env.bs_state[bs_idx, 2].item()
        used_bw = bs_bw_used[bs_idx].item()
        util_percent = (used_bw / total_bw) * 100 if total_bw > 0 else 0

        print(f"BS {bs_idx} [Position: ({env.bs_state[bs_idx, 0]:.1f}, {env.bs_state[bs_idx, 1]:.1f})]")
        print(f"  UEs Connected: {bs_ue_counts[bs_idx].item()}")
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


def visualize_network(env, bs_assignment, bandwidth_allocation):
    plt.figure(figsize=(12, 10))

    # Plot Base Stations
    bs_x = env.bs_state[:, 0].cpu().numpy()
    bs_y = env.bs_state[:, 1].cpu().numpy()
    plt.scatter(bs_x, bs_y, c='black', s=300, marker='^', label='Base Stations')

    # Plot UEs
    ue_x = env.ue_state[:, 0].cpu().numpy()
    ue_y = env.ue_state[:, 1].cpu().numpy()

    # Calculate allocation ratios for coloring
    allocation_ratios = []
    for i in range(env.ue_state.shape[0]):
        if bs_assignment[i] != -1:
            ratio = bandwidth_allocation[i].item() / env.ue_state[i, 4].item()
            allocation_ratios.append(min(ratio, 1.0))
        else:
            allocation_ratios.append(0)

    # Create a colormap: red (0) -> yellow (0.5) -> green (1)
    colors = plt.cm.rainbow([x / 2 + 0.5 for x in allocation_ratios])

    # Plot connections
    for i in range(env.ue_state.shape[0]):
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
        max_dist = env.bs_state[i, 4].item()
        circle = patches.Circle((x, y), max_dist,
                                fill=False, color='blue', alpha=0.2, linestyle='--')
        plt.gca().add_patch(circle)

        # Annotate BS with its ID
        plt.text(x, y, f"BS {i}", ha='center', va='center',
                 fontsize=10, weight='bold', color='white')

    plt.title("Network Allocation State\n"
              f"Colors show bandwidth fulfillment (red=outage, green=fully allocated)")
    plt.xlabel("X Coordinate (meters)")
    plt.ylabel("Y Coordinate (meters)")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def create_scenario(csv_path, bs_positions=[(0.0, 0.0), (10.0, 10.0), (-50.0, -50.0)],
                    max_power=None, max_bandwidth=None):
    bs_list = []
    for i, (x, y) in enumerate(bs_positions):
        bw = 100.0 + i * 20 if max_bandwidth is None else (
            max_bandwidth[i] if isinstance(max_bandwidth, list) else max_bandwidth
        )
        power = 50.0 + i * 10 if max_power is None else (
            max_power[i] if isinstance(max_power, list) else max_power
        )
        bs_list.append([x, y, bw, power, 0.0])

    bs_state = torch.tensor(bs_list, dtype=torch.float32)
    if torch.cuda.is_available():
        bs_state = bs_state.cuda()
    return ResourceEnvironment(bs_state, csv_path)


if __name__ == "__main__":
    # Create environment
    env = create_scenario(
        'ue_data.csv',
        bs_positions=[(0.0, 3000.0), (0.0, -3000.0)],
        max_bandwidth=[300, 300],
        max_power=[10, 10]
    )

    # Initialize model
    model = AllocatorNN(
        ue_input_dim=7,
        bs_input_dim=5,
        hidden_dim=128,
        num_bs=env.bs_state.shape[0]
    ).to(env.device)

    # Train the model
    train(model, env, epochs=1000)