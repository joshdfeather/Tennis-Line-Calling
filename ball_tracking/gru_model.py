import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "/Users/joshfeather/Documents/ball_tracking_protoyping/trajectory_models/tennis_trajectory_model_gru.pth"
RETRAIN = False
FREQUENCY_SCALER = 4

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

# Generate realistic trajectory
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

# Dataset class
class TrajectoryDataset(Dataset):
    def __init__(self, num_samples=20000, error_margin=0.06, frequency_scaler=FREQUENCY_SCALER):
        self.data = []
        self.error_margin = error_margin
        self.frequency_scaler = frequency_scaler
        for _ in range(num_samples):
            trajectory = generate_trajectory()
            if len(trajectory) > 1:
                landing_point = trajectory[-1]  # Last point in trajectory
                x_land, y_land, _ = landing_point
                # Select points based on FREQUENCY_SCALER
                if (
                    -COURT_LENGTH / 2 - 1 <= x_land <= COURT_LENGTH / 2 + 1 and
                    -COURT_WIDTH / 2 - 1 <= y_land <= COURT_WIDTH / 2 + 1
                ):
                    x_input = trajectory[::self.frequency_scaler]
                    # Add random uniform noise
                    noise = np.random.uniform(-self.error_margin, self.error_margin, x_input.shape)
                    x_input_noisy = x_input + noise
                    y_target = trajectory
                    # Interpolate noisy input to match full trajectory length
                    x_interp_indices = np.linspace(0, len(x_input_noisy) - 1, len(y_target))
                    x_input_interp = np.array([np.interp(x_interp_indices, np.arange(len(x_input_noisy)), x_input_noisy[:, i]) for i in range(x_input_noisy.shape[1])]).T
                    self.data.append((x_input_interp, y_target))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x_input_interp, y_target = self.data[idx]
        return (
            torch.tensor(x_input_interp, dtype=torch.float32),
            torch.tensor(y_target, dtype=torch.float32),
            len(y_target),
        )

# GRU-based Model with Dropout
class GRU_Model(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        output = self.fc(gru_out)
        return output
    
def smoothness_loss(output):
    d1 = output[:, 1:, :] - output[:, :-1, :]
    d2 = d1[:, 1:, :] - d1[:, :-1, :]
    return torch.mean(d2 ** 2)

def endpoint_loss(output, target):
    return np.mean(np.abs(output - target))

def train_model(model, dataset, num_epochs=12, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)   
    for epoch in range(num_epochs):
        total_loss = 0
        for x_seq, y_seq, y_length in dataset:
            x_seq, y_seq = x_seq.to(device), y_seq.to(device)
            optimizer.zero_grad()
            output = model(x_seq)   
            y_end = y_seq.numpy().squeeze()[-1]
            out_end = output.detach().numpy().squeeze()[-1]
            loss = criterion(output[:, :y_length, :], y_seq[:, :y_length, :])
            loss += 0.2 * (smoothness_loss(output) + endpoint_loss(out_end, y_end))
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(dataset):.6f}")

def test_model(model, error_margin=0.06):
    model.eval()
    with torch.no_grad():
        while True:
            actual_trajectory = generate_trajectory()
            landing_point = actual_trajectory[-1]  # Last point in trajectory
            x_land, y_land, _ = landing_point
            # Select points based on FREQUENCY_SCALER
            if (
                -COURT_LENGTH / 2 - 1 <= x_land <= COURT_LENGTH / 2 + 1 and
                -COURT_WIDTH / 2 - 1 <= y_land <= COURT_WIDTH / 2 + 1
            ):
                break
        erroneous_trajectory = actual_trajectory.copy()[::FREQUENCY_SCALER]
        error = np.random.uniform(-error_margin, error_margin, erroneous_trajectory.shape)
        erroneous_trajectory += error
        input_length = len(erroneous_trajectory) 
        x_interp_indices = np.linspace(0, input_length - 1, input_length * FREQUENCY_SCALER)
        x_input_interp = np.array([np.interp(x_interp_indices, np.arange(input_length), erroneous_trajectory[:, i]) for i in range(erroneous_trajectory.shape[1])]).T
        # Where to fit in pipeline 
        x_test = torch.tensor(x_input_interp, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_trajectory = model(x_test).squeeze(0).cpu().numpy()
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        # Plot actual trajectory
        ax.plot(actual_trajectory[:, 0], actual_trajectory[:, 1], actual_trajectory[:, 2],
                label="Actual Trajectory", color="b", linestyle="solid", alpha=0.7)
        # Plot predicted trajectory
        ax.plot(predicted_trajectory[3:, 0], predicted_trajectory[3:, 1], predicted_trajectory[3:, 2],
        label="Predicted Trajectory (Skipping First 5)", color="r", linestyle="--", alpha=0.7)

        # Add Tennis Court Surface
        court_x = np.array([[-COURT_LENGTH / 2, COURT_LENGTH / 2], [-COURT_LENGTH / 2, COURT_LENGTH / 2]])
        court_y = np.array([[-COURT_WIDTH / 2, -COURT_WIDTH / 2], [COURT_WIDTH / 2, COURT_WIDTH / 2]])
        court_z = np.zeros_like(court_x)
        ax.plot_surface(court_x, court_y, court_z, color='green', alpha=0.5)
        # Add Net (Vertical Plane at x=0)
        net_x = np.array([[0, 0], [0, 0]])
        net_y = np.array([[-COURT_WIDTH / 2, COURT_WIDTH / 2], [-COURT_WIDTH / 2, COURT_WIDTH / 2]])
        net_z = np.array([[0, 0], [NET_HEIGHT, NET_HEIGHT]])
        ax.plot_surface(net_x, net_y, net_z, color='black', alpha=0.8)
        # Set axis limits
        ax.set_xlim([-COURT_LENGTH / 2, COURT_LENGTH / 2])
        ax.set_ylim([-COURT_WIDTH / 2, COURT_WIDTH / 2])
        ax.set_zlim([0, 2])  # Adjust height range for visualization
        ax.set_box_aspect([COURT_LENGTH, COURT_WIDTH, 5])  # Scale X, Y, Z
        # Labels and title
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Z Position')
        ax.set_title("Actual vs Predicted 3D Trajectory")
        plt.legend()
        plt.show()

# Prepare Model
model = GRU_Model()
if os.path.exists(MODEL_PATH) and not RETRAIN:
    print("Loading pre-trained model")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    dataset = TrajectoryDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    train_model(model, dataloader)
    torch.save(model.state_dict(), MODEL_PATH)
    print("Model saved successfully")
    
test_model(model)
