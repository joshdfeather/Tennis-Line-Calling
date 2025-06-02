import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Constants
GRAVITY = 9.81
MASS = 0.0594
RADIUS = 0.0335
DRAG_COEFF = 0.507
AIR_DENSITY = 1.204
CROSS_SECTIONAL_AREA = np.pi * RADIUS**2
DT = 1 / 240
TIME_HORIZON = 4.5
NUM_STEPS = int(TIME_HORIZON / DT)
COURT_LENGTH = 23.77
COURT_WIDTH = 10.97
NET_HEIGHT = 0.914
FREQUENCY_SCALER = 4

# Models
class LSTM_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

class GRU_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        return self.fc(gru_out)

class TCN_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3, num_layers=4, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size, padding=dilation, dilation=dilation))
            layers.append(nn.ReLU())
        self.tcn = nn.Sequential(*layers)
        self.output_layer = nn.Conv1d(hidden_size, output_size, 1)
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        x = self.output_layer(x)
        return x.transpose(1, 2)

class Seq2Seq_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        _, (h_n, c_n) = self.encoder(x)
        decoder_input = torch.zeros_like(x)
        outputs, _ = self.decoder(decoder_input, (h_n, c_n))
        return self.output_layer(outputs)

class BiLSTM_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)

    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_trajectory():
    x = np.random.uniform(-COURT_LENGTH / 2, COURT_LENGTH / 2)
    y = np.random.uniform(-COURT_WIDTH / 2, COURT_WIDTH / 2)
    z = np.random.uniform(0.5, 1.5)
    spin_vector = np.random.uniform(-1, 1, 3)
    vx, vy, vz = (
        np.random.uniform(10, 30) * (-1 if x > 0 else 1),
        np.random.uniform(-5, 5),
        np.random.uniform(2, 5),
    )
    trajectory = [(x, y, z)]
    for _ in range(NUM_STEPS):
        v = np.linalg.norm([vx, vy, vz])
        drag_force = 0.5 * DRAG_COEFF * AIR_DENSITY * CROSS_SECTIONAL_AREA * v**2
        drag_acc = drag_force / MASS
        drag_vector = drag_acc * np.array([vx, vy, vz]) / (v + 1e-5)
        magnus_force = (
            0.5 * AIR_DENSITY * CROSS_SECTIONAL_AREA * v**2 * np.cross(spin_vector, [vx, vy, vz]) / (v + 1e-5)
        )
        magnus_acc = magnus_force / MASS
        ax, ay, az = (
            -drag_vector[0] + magnus_acc[0],
            -drag_vector[1] + magnus_acc[1],
            -GRAVITY - drag_vector[2] + magnus_acc[2],
        )
        vx, vy, vz = vx + ax * DT, vy + ay * DT, vz + az * DT
        x, y, z = x + vx * DT, y + vy * DT, z + vz * DT
        trajectory.append((x, y, z))
        if z <= 0:
            break
    return np.array(trajectory)

def load_sample_trajectory():
    while True:
        actual_trajectory = generate_trajectory()
        x_land, y_land, _ = actual_trajectory[-1]
        if -COURT_LENGTH / 2 - 1 <= x_land <= COURT_LENGTH / 2 + 1 and \
           -COURT_WIDTH / 2 - 1 <= y_land <= COURT_WIDTH / 2 + 1:
            break
    erroneous_trajectory = actual_trajectory.copy()[::FREQUENCY_SCALER]
    error = np.random.uniform(-0.06, 0.06, erroneous_trajectory.shape)
    erroneous_trajectory += error
    input_length = len(erroneous_trajectory)
    x_interp_indices = np.linspace(0, input_length - 1, input_length * FREQUENCY_SCALER)
    x_input_interp = np.array([
        np.interp(x_interp_indices, np.arange(input_length), erroneous_trajectory[:, i])
        for i in range(erroneous_trajectory.shape[1])
    ]).T
    return torch.tensor(x_input_interp, dtype=torch.float32).unsqueeze(0).to(device), actual_trajectory, erroneous_trajectory

def compute_errors(predicted_trajectory, actual_trajectory):
    endpoint_error = np.linalg.norm(actual_trajectory[-1][:2] - predicted_trajectory[-1][:2])
    dtw_distance, _ = fastdtw(actual_trajectory[:, :3], predicted_trajectory[:, :3], dist=euclidean)
    return endpoint_error, dtw_distance

def compute_lateral_deviation(predicted_trajectory, actual_trajectory):
    min_len = min(len(predicted_trajectory), len(actual_trajectory))
    return np.mean(np.linalg.norm(predicted_trajectory[:min_len, :2] - actual_trajectory[:min_len, :2], axis=1))

def compute_confidence_score(predicted_trajectory, actual_trajectory):
    min_len = min(len(predicted_trajectory), len(actual_trajectory))
    diff = predicted_trajectory[:min_len, :3] - actual_trajectory[:min_len, :3]
    avg_distance = np.mean(np.linalg.norm(diff, axis=1))
    return np.clip(1 / (1 + avg_distance), 0, 1)

def compute_map(predicted_trajectory, actual_trajectory, thresholds=(0.12, 0.06)):
    endpoint_error = np.linalg.norm(actual_trajectory[-1][:2] - predicted_trajectory[-1][:2])
    return [1.0 if endpoint_error <= t else 0.0 for t in thresholds]

def evaluate_model(model, num_samples=100):
    model.eval()
    endpoint_errors = []
    dtw_errors = []
    lateral_devs = []
    confidences = []
    map_50_scores = []
    map_95_scores = []

    with torch.no_grad():
        for _ in range(num_samples):
            x_test, actual_trajectory, erroneous_trajectory = load_sample_trajectory()
            predicted = model(x_test).squeeze(0).cpu().numpy()

            ep_err, dtw_err = compute_errors(predicted, actual_trajectory)
            lat_dev = compute_lateral_deviation(predicted, actual_trajectory)
            conf = compute_confidence_score(predicted, actual_trajectory)
            map50, map95 = compute_map(predicted, actual_trajectory)

            endpoint_errors.append(ep_err)
            dtw_errors.append(dtw_err)
            lateral_devs.append(lat_dev)
            confidences.append(conf)
            map_50_scores.append(map50)
            map_95_scores.append(map95)

    return (
        np.mean(endpoint_errors),
        np.mean(dtw_errors),
        np.mean(lateral_devs),
        np.mean(confidences),
        np.mean(map_50_scores),
        np.mean(map_95_scores),
    )

def evaluate_erroneous(num_samples=100):
    endpoint_errors = []
    dtw_errors = []
    lateral_devs = []
    confidences = []
    map_50_scores = []
    map_95_scores = []

    for _ in range(num_samples):
        x_test, actual, erroneous = load_sample_trajectory()
        ep_err, dtw_err = compute_errors(erroneous, actual)
        lat_dev = compute_lateral_deviation(erroneous, actual)
        conf = compute_confidence_score(erroneous, actual)
        map50, map95 = compute_map(erroneous, actual)

        endpoint_errors.append(ep_err)
        dtw_errors.append(dtw_err)
        lateral_devs.append(lat_dev)
        confidences.append(conf)
        map_50_scores.append(map50)
        map_95_scores.append(map95)

    return (
        np.mean(endpoint_errors),
        np.mean(dtw_errors),
        np.mean(lateral_devs),
        np.mean(confidences),
        np.mean(map_50_scores),
        np.mean(map_95_scores),
    )


# Evaluation
MODEL_DIR = "trajectory_models"
model_paths = [os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
model_results = []

print("\n🔍 Evaluating Erroneous Trajectory")
err_ep_err, err_dtw_err, err_lat_dev, err_conf, err_map50, err_map95 = evaluate_erroneous(num_samples=10)

for model_path in model_paths:
    print(f"\n🔍 Evaluating Model: {model_path}")
    model_path_lower = model_path.lower()
    if "bi" in model_path_lower:
        model = BiLSTM_Model().to(device)
        model_type = "BI_LSTM"
    elif "lstm" in model_path_lower:
        model = LSTM_Model().to(device)
        model_type = "LSTM"
    elif "tcn" in model_path_lower:
        model = TCN_Model().to(device)
        model_type = "TCN"
    else:
        print(f"⚠️ Skipping unknown model type: {model_path}")
        continue

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    avg_ep_err, avg_dtw_err, avg_lat_dev, avg_conf, avg_map50, avg_map95 = evaluate_model(model, num_samples=10)
    model_results.append((model_type, avg_ep_err, avg_dtw_err, avg_lat_dev, avg_conf, avg_map50, avg_map95))


# 📋 Print Summary
print("\n📊 Model Performance Summary:")
print("=" * 100)
print(f"{'Model Type':<15} {'Conf':<10} {'mAP@50':<10} {'mAP@95':<10}")
print("=" * 80)
for model_type, ep, dtw, lat, conf, map50, map95 in model_results:
    print(f"{model_type:<15} {conf:<10.4f} {map50:<10.4f} {map95:<10.4f}")
print("=" * 80)


# 📈 Prepare for Plotting
model_types = ['Erroneous'] + [model_type for model_type, avg_ep_err, avg_dtw_err, avg_lat_dev, avg_conf, avg_map50, avg_map95 in model_results]
ep_errors = [err_ep_err] + [ep for model_type, ep, avg_dtw_err, avg_lat_dev, avg_conf, avg_map50, avg_map95 in model_results]
dtw_errors = [err_dtw_err] + [dtw for model_type, avg_ep_err, dtw, avg_lat_dev, avg_conf, avg_map50, avg_map95 in model_results]
x = np.arange(len(model_types))

# Normalize separately for each metric
norm_ep = mcolors.Normalize(vmin=min(ep_errors), vmax=max(ep_errors))
norm_dtw = mcolors.Normalize(vmin=min(dtw_errors), vmax=max(dtw_errors))
cmap = cm.get_cmap('viridis')

colors_ep = [cmap(norm_ep(val)) for val in ep_errors]
colors_dtw = [cmap(norm_dtw(val)) for val in dtw_errors]

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 7.5))

# 🟦 Endpoint Error Plot
bars1 = ax1.bar(x, ep_errors, color=colors_ep)
ax1.set_title("Endpoint Error")
ax1.set_ylabel("Error Distance (m)")
ax1.set_xticks(x)
ax1.set_xticklabels(model_types, rotation=45, ha='right')
ax1.margins(x=0.1, y=0.1)  # Add internal margin
for bar, val in zip(bars1, ep_errors):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.4f}', ha='center', va='bottom')

# Add colorbar
sm_ep = cm.ScalarMappable(cmap=cmap, norm=norm_ep)
sm_ep.set_array([])
fig.colorbar(sm_ep, ax=ax1, orientation='vertical')

# 🟧 DTW Error Plot
bars2 = ax2.bar(x, dtw_errors, color=colors_dtw)
ax2.set_title("DTW Error")
ax2.set_ylabel("DTW Magnitude")
ax2.set_xticks(x)
ax2.set_xticklabels(model_types, rotation=45, ha='right')
ax2.margins(x=0.1, y=0.1) 
for bar, val in zip(bars2, dtw_errors):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.005, f'{val:.4f}', ha='center', va='bottom')

# Add colorbar
sm_dtw = cm.ScalarMappable(cmap=cmap, norm=norm_dtw)
sm_dtw.set_array([])
fig.colorbar(sm_dtw, ax=ax2, orientation='vertical')

# Layout
fig.suptitle("Trajectory Model Errors", fontsize=14, y=0.98)
fig.tight_layout()
fig.subplots_adjust(top=0.88)
plt.show()

