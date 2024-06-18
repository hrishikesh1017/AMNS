#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Function to load data from an Excel file
def load_data():
    Tk().withdraw()  # Prevent the root window from appearing
    filename = askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
    data = pd.read_excel("Vertical_Motor.xlsx")
    return data

# Load the data
data = load_data()

# Assuming the data has columns 'timestamp' and 'vibration'
timestamps = pd.to_datetime(data['timestamp'])
vibration_signal = data['vibration'].values

# Calculate the time differences in seconds
time_diff = (timestamps - timestamps[0]).total_seconds().values

# Plot the time-domain signal
plt.plot(time_diff, vibration_signal)
plt.title("Time-Domain Vibration Signal")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.show()

# FFT transformation
fs = 1 / np.mean(np.diff(time_diff))  # Calculate the sampling frequency
N = len(vibration_signal)
yf = fft(vibration_signal)
xf = fftfreq(N, 1/fs)

# Plot the frequency-domain signal
plt.plot(xf[:N//2], np.abs(yf[:N//2]))
plt.title("Frequency-Domain Vibration Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.show()

# Extract features (amplitude spectrum)
features = np.abs(yf[:N//2])

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features.reshape(-1, 1))

# Apply K-means clustering
kmeans = KMeans(n_clusters=2)
kmeans.fit(features_scaled)

# Cluster labels
labels = kmeans.labels_

# Plot the clusters
plt.scatter(xf[:N//2], features, c=labels, cmap='viridis')
plt.title("Frequency-Domain Vibration Signal with K-means Clustering")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.show()

# %%
