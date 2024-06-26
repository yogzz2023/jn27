import numpy as np
import math
import csv
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.Pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time
        self.Z = np.zeros((3, 1))  # Measurement vector

    def initialize_filter_state(self, x, y, z, vx, vy, vz, time):
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Meas_Time = time

    def predict_step(self, current_time):
        dt = current_time - self.Meas_Time
        Phi = np.eye(6)
        Phi[0, 3] = dt
        Phi[1, 4] = dt
        Phi[2, 5] = dt
        Q = np.eye(6) * self.plant_noise
        self.Sp = np.dot(Phi, self.Sf)
        self.Pp = np.dot(np.dot(Phi, self.Pf), Phi.T) + Q
        self.Meas_Time = current_time

    def update_step(self):
        Inn = self.Z - np.dot(self.H, self.Sf)  # Innovation
        S = np.dot(self.H, np.dot(self.Pf, self.H.T)) + self.R
        K = np.dot(np.dot(self.Pf, self.H.T), np.linalg.inv(S))
        self.Sf = self.Sf + np.dot(K, Inn)
        self.Pf = np.dot(np.eye(6) - np.dot(K, self.H), self.Pf)

def sph2cart(az, el, r):
    x = r * np.cos(el * np.pi / 180) * np.sin(az * np.pi / 180)
    y = r * np.cos(el * np.pi / 180) * np.cos(az * np.pi / 180)
    z = r * np.sin(el * np.pi / 180)
    return x, y, z

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

def cart2sph2(x, y, z, filtered_values_csv):
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

def read_measurements_from_csv(file_path):
    measurements = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            mr = float(row[7])  
            ma = float(row[8])  
            me = float(row[9])  
            mt = float(row[10])  
            x, y, z = sph2cart(ma, me, mr)
            r, az, el = cart2sph(x, y, z)
            measurements.append((r, az, el, mt))
    return measurements

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

def generate_hypotheses(clusters):
    hypotheses = []
    for cluster in clusters:
        if len(cluster) == 1:
            hypotheses.append(cluster)
        else:
            for measurement in cluster:
                hypotheses.append([measurement])
    return hypotheses

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

kalman_filter = CVFilter()
csv_file_path = 'ttk_84_2.csv'

measurements = read_measurements_from_csv(csv_file_path)
measurement_groups = form_measurement_groups(measurements)

csv_file_predicted = "ttk_84_2.csv"
df_predicted = pd.read_csv(csv_file_predicted)
filtered_values_csv = df_predicted[['F_TIM', 'F_X', 'F_Y', 'F_Z']].values

A = cart2sph2(filtered_values_csv[:,1], filtered_values_csv[:,2], filtered_values_csv[:,3], filtered_values_csv)
number = 1000
result = np.divide(A[0], number)

time_list = []
r_list = []
az_list = []
el_list = []

# Process the first two measurements to initialize the filter and compute velocities
initial_measurements = measurement_groups[0]
if len(initial_measurements) >= 2:
    r1, az1, el1, t1 = initial_measurements[0]
    r2, az2, el2, t2 = initial_measurements[1]
    dt = t2 - t1
    if dt != 0:
        vx = (r2 - r1) / dt
        vy = (az2 - az1) / dt
        vz = (el2 - el1) / dt
    else:
        vx, vy, vz = 0, 0, 0
    
    kalman_filter.initialize_filter_state(r2, az2, el2, vx, vy, vz, t2)

# Process the rest of the measurements starting from the third one
for i in range(1, len(measurement_groups)):
    group = measurement_groups[i]
    clusters = form_clusters(group)
    hypotheses = generate_hypotheses(clusters)
    joint_probabilities = calculate_joint_probabilities(hypotheses, kalman_filter)

    best_hypothesis = hypotheses[np.argmax(joint_probabilities)]

    for j, (r, az, el, mt) in enumerate(best_hypothesis):
        if i == 1 and j == 0:
            # Skip the first group as it's used for initialization
            continue
        kalman_filter.predict_step(mt)
        kalman_filter.Z = np.array([[r], [az], [el]])
        kalman_filter.update_step()

        time_list.append(mt)
        r_list.append(r)
        az_list.append(az)
        el_list.append(el)

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
