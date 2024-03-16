import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

FILENAME_E4E16 = os.path.join(os.path.dirname(__file__), 'e4e16.txt')
FILENAME_E0E24 = os.path.join(os.path.dirname(__file__), 'e0e24.txt')

with open(FILENAME_E4E16, 'r') as file_e4e16:
    loss_e4e16 = []
    epoch_e4e16 = []
    lr_e4e16 = []
    for line in file_e4e16.readlines():
        try:
            temp_json = json.loads(line.strip())
            loss_e4e16.append(float(temp_json['loss']))
            epoch_e4e16.append(float(temp_json['epoch']))
            lr_e4e16.append(float(temp_json['learning_rate']))
        except:
            continue

with open(FILENAME_E0E24, 'r') as file_e4e16:
    loss_e0e24 = []
    epoch_e0e24 = []
    lr_e0e24 = []
    for line in file_e4e16.readlines():
        try:
            temp_json = json.loads(line.strip())
            loss_e0e24.append(float(temp_json['loss']))
            epoch_e0e24.append(float(temp_json['epoch']))
            lr_e0e24.append(float(temp_json['learning_rate']))
        except:
            continue

plt.plot(epoch_e4e16, loss_e4e16, color='red', ls='--', label='E4E16 Loss - With Language Learning')
plt.plot(epoch_e0e24, loss_e0e24, color='blue', ls=':', label='E0E24 Loss - Without Language Learning')
plt.title("OUR Tokenize Model Loss Comparision")
plt.yscale("log")
plt.legend()
plt.show()
plt.figure()
plt.plot(epoch_e4e16, lr_e4e16, color='red', ls='--',label='E4E16 Learning Rate - With Language Learning')
plt.plot(epoch_e0e24, lr_e0e24, color='blue', ls=':',label='E0E24 Learning Rate - Without Language Learning')
plt.title("OUR Tokenize Model Showcase")
plt.legend()
plt.show()