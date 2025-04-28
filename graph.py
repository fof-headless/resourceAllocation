import matplotlib.pyplot as plt

# Number of UEs
ues = [30, 60, 90, 120, 150]

# Execution times
ml_times = [0.06, 0.06, 0.06, 0.06, 0.06]
rule_based_times = [1.0, 1.0, 1.0, 1.0, 1.25]

# Create the plot
plt.figure(figsize=(8, 5))
plt.plot(ues, ml_times, marker='o', color='blue', label='ML-based Allocation', linewidth=2)
plt.plot(ues, rule_based_times, marker='s', color='red', label='Rule-based Allocation', linewidth=2)

# Add labels and title
plt.xlabel('Number of UEs')
plt.ylabel('Execution Time (ms)')
plt.title('Execution Time Comparison: ML-based vs Rule-based Allocation')
plt.grid(True)
plt.legend()
plt.ylim(0, 1.5)
plt.xticks(ues)
plt.yticks([0, 0.5, 1.0, 1.5])

# Save the figure
plt.savefig('execution_time_comparison.png', dpi=300)

# Show the plot
plt.show()