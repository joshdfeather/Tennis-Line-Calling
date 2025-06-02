import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

# --- Constants ---
FREQUENCY_SCALER = 4
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
MODEL_PATH = "/Users/joshfeather/Documents/ball_tracking_protoyping/trajectory_models/tennis_trajectory_model_bi_lstm.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model ---
class BiLSTM_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, input_size)
    def forward(self, x):
        out, _ = self.bilstm(x)
        return self.fc(out)

# --- Generate Physics-based Trajectory ---
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
            0.5 * AIR_DENSITY * CROSS_SECTIONAL_AREA * v**2 *
            np.cross(spin_vector, [vx, vy, vz]) / (v + 1e-5)
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

# --- Predict with Noise + Occlusion ---
def predict_on_trajectory(model, base_traj, error_margin, occlusion_percent):
    # Step 1: Downsample + apply noise
    noisy_input = base_traj.copy()[::FREQUENCY_SCALER]
    noise = np.random.uniform(-error_margin, error_margin, noisy_input.shape)
    noisy_input += noise
    total_points = noisy_input.shape[0]
    num_occlude = int(total_points * occlusion_percent)
    occlude_indices = np.random.choice(total_points, num_occlude, replace=False)
    occlude_mask = np.zeros(total_points, dtype=bool)
    occlude_mask[occlude_indices] = True
    # Step 2: Replace each occluded point with the next non-occluded one
    for idx in occlude_indices:
        replacement = None
        # Look forward
        for j in range(idx + 1, total_points):
            if not occlude_mask[j]:
                replacement = noisy_input[j]
                break
        # If no future valid point found, look backward
        if replacement is None:
            for j in range(idx - 1, -1, -1):
                if not occlude_mask[j]:
                    replacement = noisy_input[j]
                    break
        # Final fallback: use the same point (shouldn't happen unless 100% occlusion)
        if replacement is None:
            replacement = noisy_input[idx]
        noisy_input[idx] = replacement
    # Step 3: Interpolate for full sequence
    input_len = len(noisy_input)
    interp_indices = np.linspace(0, input_len - 1, input_len * FREQUENCY_SCALER)
    x_input_interp = np.array([
        np.interp(interp_indices, np.arange(input_len), noisy_input[:, i]) for i in range(3)
    ]).T
    x_tensor = torch.tensor(x_input_interp, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(x_tensor).squeeze(0).cpu().numpy()
    return noisy_input, output

# --- 3D Plot Function ---
def plot_3d_trajectory(error_points, predicted):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(error_points[:, 0], error_points[:, 1], error_points[:, 2], color='black', label='Noisy/Occluded Input', s=10)
    ax.plot(predicted[:, 0], predicted[:, 1], predicted[:, 2], color='red', label='Predicted Trajectory', linewidth=2)

    # Tennis court surface
    court_x = np.array([[-COURT_LENGTH / 2, COURT_LENGTH / 2], [-COURT_LENGTH / 2, COURT_LENGTH / 2]])
    court_y = np.array([[-COURT_WIDTH / 2, -COURT_WIDTH / 2], [COURT_WIDTH / 2, COURT_WIDTH / 2]])
    court_z = np.zeros_like(court_x)
    ax.plot_surface(court_x, court_y, court_z, color='green', alpha=0.3)

    # Net
    net_x = np.array([[0, 0], [0, 0]])
    net_y = np.array([[-COURT_WIDTH / 2, COURT_WIDTH / 2], [-COURT_WIDTH / 2, COURT_WIDTH / 2]])
    net_z = np.array([[0, 0], [NET_HEIGHT, NET_HEIGHT]])
    ax.plot_surface(net_x, net_y, net_z, color='black', alpha=0.6)

    ax.set_xlim([-COURT_LENGTH / 2, COURT_LENGTH / 2])
    ax.set_ylim([-COURT_WIDTH / 2, COURT_WIDTH / 2])
    ax.set_zlim([0, 2])
    ax.set_box_aspect([COURT_LENGTH, COURT_WIDTH, 6])  # Scale X, Y, Z
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title("Bi-LSTM: Noisy & Occluded Input vs Predicted Trajectory")
    ax.legend()
    return fig

# --- Streamlit App ---
st.title("Trajectory Prediction with Noise & Occlusion")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: white;
        }
        [data-testid="stHeader"], [data-testid="stToolbar"] {
            background: white;
        }
        .st-emotion-cache-1v0mbdj {
            background-color: white;
        }
        * {
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        body, div, h1, h2, h3, h4, h5, h6, p, label, .css-10trblm, .css-1d391kg {
            font-size: 25px !important;
        }

        .stSlider > div {
            font-size: 25px !important;
        }

        .st-emotion-cache-1v0mbdj, .st-emotion-cache-1v0mbdj * {
            font-size: 25px !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_data
def get_fixed_trajectory():
    while True:
        traj = generate_trajectory()
        x_land, y_land, _ = traj[-1]
        if -COURT_LENGTH / 2 - 1 <= x_land <= COURT_LENGTH / 2 + 1 and -COURT_WIDTH / 2 - 1 <= y_land <= COURT_WIDTH / 2 + 1:
            return traj

# Load model
model = BiLSTM_Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)

# Sliders
error_margin = st.slider("Noise magnitude (meters)", 0.00, 0.15, 0.06, 0.01)
occlusion_percent = st.slider("Occlusion Percentage (%)", 0, 80, 0, 5) / 100

# Run prediction
base_trajectory = get_fixed_trajectory()
error_points, predicted_trajectory = predict_on_trajectory(model, base_trajectory, error_margin, occlusion_percent)

# Plot
fig = plot_3d_trajectory(error_points, predicted_trajectory)
st.pyplot(fig)
