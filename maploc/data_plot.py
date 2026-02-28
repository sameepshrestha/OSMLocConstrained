import matplotlib.pyplot as plt

steps = [60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000]

# Training losses
train_total = [11.3, 11.6, 13.5, 12.8, 12.9, 11.8, 12.6, 13.6, 13.2, 11.7, 12.5]
train_nll   = [11.3, 11.6, 13.5, 12.8, 12.9, 11.8, 12.6, 13.6, 13.2, 11.7, 12.5]

# Validation losses
val_total = [12.43, 12.43, 12.37, 12.37, 12.32, 12.32, 12.25, 12.25, 12.22, 12.19, 12.22]
val_nll   = [12.43, 12.43, 12.37, 12.37, 12.32, 12.32, 12.25, 12.25, 12.22, 12.19, 12.22]

# XY metrics
val_xy_max_error = [38.04, 38.4, 36.64, 36.64, 36.06, 36.06, 36.06, 36.06, 36.68, 37.22, 37.22]
val_xy_recall_1m = [0.00387, 0.007186, 0.006081, 0.006081, 0.008292, 0.008292, 0.008292, 0.008292, 0.006633, 0.0105, 0.0105]
val_xy_recall_3m = [0.03925, 0.05086, 0.04809, 0.04809, 0.0503, 0.0503, 0.0503, 0.0503, 0.04533, 0.0503, 0.0503]
val_xy_recall_5m = [0.09176, 0.09121, 0.1017, 0.1017, 0.1012, 0.1012, 0.1012, 0.1012, 0.09619, 0.1083, 0.1083]

# Yaw metrics
val_yaw_max_error = [65.74, 65.65, 68.41, 68.41, 66.43, 66.43, 66.43, 66.43, 66.58, 64.16, 64.16]
val_yaw_recall_1 = [0.05473, 0.05694, 0.05252, 0.05252, 0.05804, 0.05804, 0.05804, 0.05804, 0.05417, 0.04975, 0.04975]
val_yaw_recall_3 = [0.1564, 0.1636, 0.1465, 0.1465, 0.1531, 0.1531, 0.1531, 0.1531, 0.1493, 0.1520, 0.1520]
val_yaw_recall_5 = [0.23, 0.2416, 0.2244, 0.2244, 0.2388, 0.2388, 0.2388, 0.2388, 0.2261, 0.2305, 0.2305]

plt.figure(figsize=(18,5))  

# Losses
plt.subplot(1,3,1)
plt.plot(steps, train_total, label="Train Total", marker='o')
plt.plot(steps, train_nll, label="Train NLL", marker='o')
plt.plot(steps, val_total, label="Val Total", marker='x')
plt.plot(steps, val_nll, label="Val NLL", marker='x')
plt.xlabel("Global Step")
plt.ylabel("Loss")
plt.title("Losses")
plt.legend()
plt.grid(True)

# XY metrics
plt.subplot(1,3,2)
plt.plot(steps, val_xy_max_error, label="XY Max Error", marker='x')
plt.plot(steps, val_xy_recall_1m, label="XY Recall 1m", marker='x')
plt.plot(steps, val_xy_recall_3m, label="XY Recall 3m", marker='x')
plt.plot(steps, val_xy_recall_5m, label="XY Recall 5m", marker='x')
plt.xlabel("Global Step")
plt.ylabel("XY Metric")
plt.title("XY Metrics")
plt.legend()
plt.grid(True)

# Yaw metrics
plt.subplot(1,3,3)
plt.plot(steps, val_yaw_max_error, label="Yaw Max Error", marker='x')
plt.plot(steps, val_yaw_recall_1, label="Yaw Recall 1°", marker='x')
plt.plot(steps, val_yaw_recall_3, label="Yaw Recall 3°", marker='x')
plt.plot(steps, val_yaw_recall_5, label="Yaw Recall 5°", marker='x')
plt.xlabel("Global Step")
plt.ylabel("Yaw Metric")
plt.title("Yaw Metrics")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
