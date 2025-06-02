import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Constants
GRAVITY = 9.81
MASS = 0.0594
RADIUS = 0.0335
DRAG_COEFF = 0.507
AIR_DENSITY = 1.204
CROSS_SECTIONAL_AREA = np.pi * RADIUS**2
DT = 1 / 60  # Time step (60 FPS)
TIME_HORIZON_1 = 4.0  # Total duration for full trajectory (seconds)
NUM_STEPS_1 = int(TIME_HORIZON_1 / DT)  # Steps per full trajectory
TIME_HORIZON_2 = 2.0  # Trimmed trajectory duration (seconds)
TRIMMED_LENGTH = int(TIME_HORIZON_2 / DT)  # Steps per trimmed trajectory
COURT_LENGTH = 23.77  # Full length of a tennis court
COURT_WIDTH = 10.97  # Doubles court width
NET_HEIGHT = 0.914  # Net height in meters
WINDOW_SIZE = 4

def generate_trajectory():
    """Generate a physics-based trajectory for a tennis shot."""
    x = np.random.uniform(-COURT_LENGTH / 2, COURT_LENGTH / 2)
    y = np.random.uniform(-COURT_WIDTH / 2, COURT_WIDTH / 2)
    z = np.random.uniform(0.5, 1.5)
    
    spin_vector = [np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]
    
    vx = np.random.uniform(10, 30) * (-1 if x > 0 else 1)
    vy = np.random.uniform(-5, 5)
    vz = np.random.uniform(2, 5)
    
    trajectory = [(x, y, z)]
    
    for _ in range(NUM_STEPS_1):
        v = np.linalg.norm([vx, vy, vz])
        drag_force = 0.5 * DRAG_COEFF * AIR_DENSITY * CROSS_SECTIONAL_AREA * v**2
        drag_acc = drag_force / MASS
        drag_vector = drag_acc * np.array([vx, vy, vz]) / v

        magnus_force = (
            0.5 * AIR_DENSITY * CROSS_SECTIONAL_AREA * v**2 *
            np.cross(spin_vector, [vx, vy, vz]) / v
        )
        magnus_acc = magnus_force / MASS

        ax = -drag_vector[0] + magnus_acc[0]
        ay = -drag_vector[1] + magnus_acc[1]
        az = -GRAVITY - drag_vector[2] + magnus_acc[2]

        vx += ax * DT
        vy += ay * DT
        vz += az * DT

        x += vx * DT
        y += vy * DT
        z += vz * DT

        trajectory.append((x, y, z))
    
    return np.array(trajectory)

def generate_data(num_samples=10000):
    data = []
    for _ in range(num_samples):
        traj = generate_trajectory()
        valid_indices = [i for i in range(len(traj) - TRIMMED_LENGTH) if traj[i, 2] > 0]
        if valid_indices:
            i = np.random.choice(valid_indices)
            trimmed_traj = traj[i:i+TRIMMED_LENGTH]
            data.append(trimmed_traj)
    data = np.array(data)
    X_train = data[:, :WINDOW_SIZE, :]
    y_train = data
    return torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)

# Generate dataset
X_train, y_train = generate_data(num_samples=10000)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, y_train = X_train.to(device), y_train.to(device)

class EncoderLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2):
        super(EncoderLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden, cell) = self.lstm(x)
        return hidden, cell

class DecoderLSTM(nn.Module):
    def __init__(self, output_size=3, hidden_size=64, num_layers=2):
        super(DecoderLSTM, self).__init__()
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell, num_steps):
        outputs = []
        decoder_input = x  
        
        for _ in range(num_steps):
            lstm_out, (hidden, cell) = self.lstm(decoder_input, (hidden, cell))
            out = self.fc(lstm_out[:, -1, :]).unsqueeze(1)
            decoder_input = out  
            outputs.append(out)
        
        return torch.cat(outputs, dim=1)

class Seq2SeqModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=3):
        super(Seq2SeqModel, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(output_size, hidden_size, num_layers)
    
    def forward(self, x, num_steps):
        hidden, cell = self.encoder(x)
        predictions = self.decoder(x[:, -1:, :], hidden, cell, num_steps)
        return predictions

# Initialize model, loss function, and optimizer
model = Seq2SeqModel().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
batch_size = 64
num_epochs = 50

for epoch in range(num_epochs):
    total_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size].to(device)
        y_batch = y_train[i:i+batch_size].to(device)
        optimizer.zero_grad()
        predictions = model(X_batch, TRIMMED_LENGTH)
        loss = criterion(predictions[:, 4:, :], y_batch[:, 4:, :])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / (len(X_train) // batch_size)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Test model with sliding window approach
def test_model(traj):
    predicted_traj = []
    for i in range(len(traj) - 4):
        input_seq = torch.tensor(traj[i:i+4], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted = model(input_seq, TRIMMED_LENGTH).cpu().numpy()[0]
        predicted_traj.append(predicted[0])
    
    predicted_traj = np.array(predicted_traj)
    predicted_traj = predicted_traj[predicted_traj[:, 2] > 0]  # Stop when z < 0
    return predicted_traj

# Example test trajectory
test_traj = generate_trajectory()
predicted_traj = test_model(test_traj)
actual_traj = test_traj[test_traj[:, 2] > 0]

# 3D Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Actual trajectory (x, y, z)
ax.plot(actual_traj[:, 0], actual_traj[:, 1], actual_traj[:, 2], label="Actual Trajectory", color="b", marker='o')

# Predicted trajectory (x, y, z)
ax.plot(predicted_traj[:, 0], predicted_traj[:, 1], predicted_traj[:, 2], label="Predicted Trajectory", color="r", linestyle="--", marker='x')

# Add Tennis Court Surface
court_x = np.array([[-COURT_LENGTH / 2, COURT_LENGTH / 2], [-COURT_LENGTH / 2, COURT_LENGTH / 2]])
court_y = np.array([[-COURT_WIDTH / 2, -COURT_WIDTH / 2], [COURT_WIDTH / 2, COURT_WIDTH / 2]])
court_z = np.zeros_like(court_x)  # Court is at z = 0
ax.plot_surface(court_x, court_y, court_z, color='green', alpha=0.5)

# Add Net (Vertical Plane at x=0)
net_x = np.array([[0, 0], [0, 0]])
net_y = np.array([[-COURT_WIDTH / 2, COURT_WIDTH / 2], [-COURT_WIDTH / 2, COURT_WIDTH / 2]])
net_z = np.array([[0, 0], [NET_HEIGHT, NET_HEIGHT]])
ax.plot_surface(net_x, net_y, net_z, color='black', alpha=0.6)

# Restrict plot limits to court dimensions
ax.set_xlim([-COURT_LENGTH/ 2, COURT_LENGTH/ 2])
ax.set_ylim([-COURT_WIDTH / 2, COURT_WIDTH / 2])
ax.set_zlim([0, 2])  # Set a reasonable height limit for better visualization
ax.set_box_aspect([COURT_LENGTH, COURT_WIDTH, 4])  # Scale X, Y, Z

# Labels and title
ax.set_xlabel('X Position')
ax.set_ylabel('Y Position')
ax.set_zlabel('Z Position')
ax.set_title("Actual vs Predicted 3D Trajectory")
ax.legend()
plt.show() 