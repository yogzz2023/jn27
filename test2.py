import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

# Define your CVFilter class
class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))  # Measurement vector

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def predict_step(self, current_time):
        # Predict step
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q

    def update_step(self):
        # Update step with measurement
        Inn = self.Z - np.dot(self.H, self.Sf)  # Innovation
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

# Function to convert spherical coordinates to Cartesian coordinates
def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

# Function to convert Cartesian coordinates to spherical coordinates
def cart2sph(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    el = math.atan(z / np.sqrt(x**2 + y**2)) * 180 / 3.14
    az = math.atan(y / x)

    if x > 0.0:
        az = 3.14 / 2 - az
    else:
        az = 3 * 3.14 / 2 - az

    az = az * 180 / 3.14

    if az < 0.0:
        az = 360 + az

    if az > 360:
        az = az - 360

    return r, az, el

def cart2sph2(x: float, y: float, z: float, filtered_values_csv):
    r = []
    az = []
    el = []
    for i in range(len(filtered_values_csv)):
        r.append(np.sqrt(x[i]**2 + y[i]**2 + z[i]**2))
        el.append(math.atan(z[i] / np.sqrt(x[i]**2 + y[i]**2)) * 180 / 3.14)
        az.append(math.atan(y[i] / x[i]))

        if x[i] > 0.0:
            az[i] = 3.14 / 2 - az[i]
        else:
            az[i] = 3 * 3.14 / 2 - az[i]

        az[i] = az[i] * 180 / 3.14

        if az[i] < 0.0:
            az[i] = 360 + az[i]

        if az[i] > 360:
            az[i] = az[i] - 360

    return r, az, el

# Function to read measurements from CSV file
def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header if exists
        for row in reader:
            mr = float(row[7])  # MR column
            ma = float(row[8])  # MA column
            me = float(row[9])  # ME column
            mt = float(row[10])  # MT column
            x, y, z = sph2cart(ma, me, mr)  # Convert spherical to Cartesian coordinates
            r, az, el = cart2sph(x, y, z)  # Convert Cartesian to spherical coordinates
            measurements.append((r, az, el, mt))
    return measurements

# Function to form measurement groups based on time
def form_measurement_groups(measurements, time_threshold=0.05):
    groups = []
    current_group = []
    for i in range(len(measurements)):
        if i == 0 or (measurements[i][3] - measurements[i-1][3]) <= time_threshold:
            current_group.append(measurements[i])
        else:
            groups.append(current_group)
            current_group = [measurements[i]]
    if current_group:
        groups.append(current_group)
    return groups

# Function to form clusters for each group using chi-square test
def form_clusters(group, threshold=0.95):
    clusters = []
    for i, measurement in enumerate(group):
        added = False
        for cluster in clusters:
            distances = [np.linalg.norm(np.array(measurement[:3]) - np.array(m[:3])) for m in cluster]
            if all(d < chi2.ppf(threshold, df=3) for d in distances):
                cluster.append(measurement)
                added = True
                break
        if not added:
            clusters.append([measurement])
    return clusters

# Function to generate hypotheses for clusters
def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        if len(cluster) == 1:
            hypotheses.append(cluster)
        else:
            for measurement in cluster:
                hypotheses.append([measurement])
    return hypotheses

# Function to calculate joint probabilities
def calculate_joint_probabilities(hypotheses, kalman_filter):
    joint_probabilities = []
    for hypothesis in hypotheses:
        likelihoods = []
        for measurement in hypothesis:
            innovation = np.array(measurement[:3]) - kalman_filter.Sf[:3, 0]
            S = kalman_filter.H @ kalman_filter.Pf @ kalman_filter.H.T + kalman_filter.R
            likelihood = np.exp(-0.5 * np.dot(innovation.T, np.linalg.inv(S) @ innovation))
            likelihoods.append(likelihood)
        joint_probabilities.append(np.prod(likelihoods))
    return joint_probabilities

# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define the path to your CSV file containing measurements
csv_file_path = 'ttk_84_2.csv'  # Provide the path to your CSV file

# Read measurements from CSV file
measurements = read_measurements_from_csv(csv_file_path)

# Form measurement groups based on time
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_84_2.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

A = cart2sph2(filtered_values_csv[:,1], filtered_values_csv[:,2], filtered_values_csv[:,3], filtered_values_csv)

number = 1000
result = np.divide(A[0], number)

# Lists to store the data for plotting
time_list = []
r_list = []
az_list = []
el_list = []

# Iterate through measurement groups
for i, group in enumerate(measurement_groups):
    clusters = form_clusters(group)
    hypotheses = generate_hypotheses(clusters)
    joint_probabilities = calculate_joint_probabilities(hypotheses, kalman_filter)

    # Select the hypothesis with the highest joint probability
    best_hypothesis = hypotheses[np.argmax(joint_probabilities)]

    # Update filter state with the best hypothesis
    for j, (r, az, el, mt) in enumerate(best_hypothesis):
        if i == 0 and j == 0:
            # Initialize filter state with the first measurement
            kalman_filter.initialize_filter_state(r, az, el, 0, 0, 0, mt)
        elif i == 1 and j == 0:
            # Initialize filter state with the second measurement and compute velocity
            prev_r, prev_az, prev_el = best_hypothesis[j-1][:3]
            dt = mt - best_hypothesis[j-1][3]
            if dt != 0:
                vx = (r - prev_r) / dt
                vy = (az - prev_az) / dt
                vz = (el - prev_el) / dt
            else:
                vx, vy, vz = 0, 0, 0  # Set velocities to 0 if dt is zero
            kalman_filter.initialize_filter_state(r, az, el, vx, vy, vz, mt)
        else:
            kalman_filter.predict_step(mt)
            kalman_filter.Z = np.array([[r], [az], [el]])  # Update measurement
            kalman_filter.update_step()

            # Append data for plotting from the third measurement onwards
            if i > 1 or (i == 1 and j > 0):
                time_list.append(mt)
                r_list.append(r)
                az_list.append(az)
                el_list.append(el)

# Plot range (r) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, r_list, color='green', label='Filtered Range (Kalman Filter)')
plt.scatter(filtered_values_csv[:, 0], result, label='filtered range (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Range (r)')
plt.title('Range vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot azimuth (az) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, az_list, color='green', label='Filtered Azimuth (Kalman Filter)')
plt.scatter(filtered_values_csv[:, 0], A[1], label='filtered azimuth (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Azimuth (az)')
plt.title('Azimuth vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot elevation (el) vs. time
plt.figure(figsize=(12, 6))
plt.scatter(time_list, el_list, color='green', label='Filtered Elevation (Kalman Filter)')
plt.scatter(filtered_values_csv[:, 0], A[2], label='filtered elevation (track id 31)', color='red', marker='*')
plt.xlabel('Time')
plt.ylabel('Elevation (el)')
plt.title('Elevation vs. Time')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
