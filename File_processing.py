import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import pywt
from scipy import signal

label = pd.read_csv("new_labels.csv", header=0, names=('patient', 'header file', 'diagnosis'))
patient_array = np.array(label['patient'])
original_file_array = np.array(label['header file'])
diagnosis_array = np.array(label['diagnosis'])


def baseline_corr(array):
    # Function for use in baseline removal
    def wrcoef(X, coef_type, coef, wavename, level):
        N = np.array(X).size
        a, ds = coef[0], list(reversed(coef[1:]))

        if coef_type == 'a':
            return pywt.upcoef('a', a, wavename, level=level)[:N]
        elif coef_type == 'd':
            return pywt.upcoef('d', ds[level - 1], wavename, level=level)[:N]
        else:
            raise ValueError("Invalid coefficient type: {}".format(coef_type))

    # Baseline correction variables
    waveName = 'db1'
    # Baseline correction
    coefficient = pywt.wavedec(array, waveName, level=10)
    # A10 = wrcoef(array, 'a', coefficient, waveName, 10)
    D10 = wrcoef(array, 'd', coefficient, waveName, 10)
    D9 = wrcoef(array, 'd', coefficient, waveName, 9)
    D8 = wrcoef(array, 'd', coefficient, waveName, 8)
    D7 = wrcoef(array, 'd', coefficient, waveName, 7)
    D6 = wrcoef(array, 'd', coefficient, waveName, 6)
    D5 = wrcoef(array, 'd', coefficient, waveName, 5)
    D4 = wrcoef(array, 'd', coefficient, waveName, 4)
    D3 = wrcoef(array, 'd', coefficient, waveName, 3)
    D2 = wrcoef(array, 'd', coefficient, waveName, 2)
    D1 = wrcoef(array, 'd', coefficient, waveName, 1)
    array = D10 + D9 + D8 + D7 + D6 + D5 + D4 + D3 + D2 + D1
    return array


def wavelet_denoise(dataframe, wavelet='sym7', level=4, mode='soft'):  # Update
    coefficients = pywt.wavedec(dataframe, wavelet, level=level)
    threshold = np.sqrt(2 * np.log(len(dataframe))) * np.median(np.abs(coefficients[-level])) / 0.6745
    coefficients[1:] = [pywt.threshold(c, threshold, mode=mode) for c in coefficients[1:]]
    return pywt.waverec(coefficients, wavelet)


# Loop through each file in the database
# patient = []
# diagnosis = []
# original_file = []
# signal_num = 0
# start_time = 0
# if os.path.exists('PTB_processed'):
#     shutil.rmtree('PTB_processed')
# os.makedirs('PTB_processed')
# for i in range(len(patient_array)):  # For each patient
#     if patient_array[i] < 10:
#         p_string = '00' + str(patient_array[i])
#     elif patient_array[i] < 100:
#         p_string = '0' + str(patient_array[i])
#     else:
#         p_string = str(patient_array[i])
#     # p_number = 'patient' + p_string
#     filepath = "PTB_original/patient" + p_string
#     if os.path.isdir(filepath):  # Check if patient folder exists
#         for filename in os.listdir(filepath):  # For each file in patient, load file
#             start_time = time.time()
#             data = pd.read_csv(
#                 filepath + '/' + filename,
#                 header=0,
#                 skiprows=1,
#                 usecols=list(range(1, 13)),
#                 engine='python'
#             )
#             patient.append(i + 1)
#             original_file.append(filename)
#             diagnosis.append(diagnosis_array[i])
#             signal_num += 1
#             if signal_num < 10:
#                 mod_signal = '00' + str(signal_num)
#             elif signal_num < 100:
#                 mod_signal = '0' + str(signal_num)
#             else:
#                 mod_signal = str(signal_num)
#             file_array = []
#             for column in data:  # Processing
#                 df = np.array(data[column])
#                 df = baseline_corr(df)
#                 df = wavelet_denoise(df)  # Updated
#                 df = df[0:30000]
#                 df = signal.resample(df, 7500)
#                 file_array.append(df)
#             newdata = pd.concat([pd.Series(file_array[0]),
#                                  pd.Series(file_array[1]),
#                                  pd.Series(file_array[2]),
#                                  pd.Series(file_array[3]),
#                                  pd.Series(file_array[4]),
#                                  pd.Series(file_array[5]),
#                                  pd.Series(file_array[6]),
#                                  pd.Series(file_array[7]),
#                                  pd.Series(file_array[8]),
#                                  pd.Series(file_array[9]),
#                                  pd.Series(file_array[10]),
#                                  pd.Series(file_array[11]), ], axis=1)
#             newdata.columns = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
#             newdata.to_csv(
#                 'PTB_processed/record' + mod_signal + '_' + p_string + '_' + str(diagnosis_array[i]) + '.csv',
#                 index=False)
#     completion = (i / len(patient_array)) * 100
#     end_time = time.time()
#     elapsed_time = round(end_time - start_time, 2)
#     sys.stdout.write(
#         '\r' + p_string + ' loaded - ' + str(completion) + '% completed' + ' Took : ' + str(elapsed_time) + ' seconds')
#
# label_df = pd.concat([pd.Series(patient), pd.Series(original_file), pd.Series(diagnosis)], axis=1)
# label_df.columns = ['patient', 'original_file', 'diagnosis']
# label_df.to_csv('labels.csv')

# print('')
# print('Done: ' + str(signal_num) + ' signals processed and loaded')
# print(get_python_lib())

patient = []
diagnosis = []
original_file = []
signal_num = 0
skipped = 0
filepath = "ptb-diagnostic-ecg-database-1.0.0"
if os.path.exists('PTB_processed_new'):
    shutil.rmtree('PTB_processed_new')
os.makedirs('PTB_processed_new')
for i in range(len(patient_array)):  # For each line in new_labels
    p_string = str(patient_array[i])
    f_string = str(original_file_array[i])
    full_filepath = filepath + '/' + p_string + '/' + f_string + '.csv'
    if os.path.exists(full_filepath):
        data = pd.read_csv(
            full_filepath,
            header=0,
            skiprows=1,
            usecols=list(range(1, 13)),
            engine='python'
        )
        start_time = time.time()
        patient.append(p_string)
        original_file.append(f_string)
        diagnosis.append(diagnosis_array[i])
        signal_num += 1
        if signal_num < 10:
            mod_signal = '00' + str(signal_num)
        elif signal_num < 100:
            mod_signal = '0' + str(signal_num)
        else:
            mod_signal = str(signal_num)
        file_array = []
        for column in data:  # Processing
            df = np.array(data[column])
            df = baseline_corr(df)
            df = wavelet_denoise(df)  # Updated
            df = df[0:30000]
            df = signal.resample(df, 7500)
            file_array.append(df)
        newdata = pd.concat([pd.Series(file_array[0]),
                             pd.Series(file_array[1]),
                             pd.Series(file_array[2]),
                             pd.Series(file_array[3]),
                             pd.Series(file_array[4]),
                             pd.Series(file_array[5]),
                             pd.Series(file_array[6]),
                             pd.Series(file_array[7]),
                             pd.Series(file_array[8]),
                             pd.Series(file_array[9]),
                             pd.Series(file_array[10]),
                             pd.Series(file_array[11]), ], axis=1)
        newdata.columns = ['i', 'ii', 'iii', 'avr', 'avl', 'avf', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6']
        newdata.to_csv(
            'PTB_processed_new/record' + mod_signal + '_' + p_string + '_' + str(diagnosis_array[i]) + '.csv',
            index=False)
        completion = (i / len(patient_array)) * 100
        end_time = time.time()
        elapsed_time = round(end_time - start_time, 2)
        sys.stdout.write('\r' + p_string + ' loaded - ' + str(completion) + '% completed' + ' Took : ' + str(
            elapsed_time) + ' seconds')
    else:
        print("Skipped " + p_string + " file " + f_string)
        skipped += 1

label_df = pd.concat([pd.Series(patient), pd.Series(original_file), pd.Series(diagnosis)], axis=1)
label_df.columns = ['patient', 'original_file', 'diagnosis']
label_df.to_csv('new_labels_processed.csv')

print('')
print('Done: ' + str(signal_num) + ' signals processed and loaded ' + str(skipped) + " skipped")
