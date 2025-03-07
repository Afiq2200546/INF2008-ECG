import os
import csv

def extract_label(header_file):
    """Extracts the reason for admission from the header file."""
    with open(header_file, 'r') as file:
        for line in file:
            if line.startswith("# Reason for admission:"):
                return line.split(":", 1)[1].strip()
    return "Unknown"

def process_ptb_directory(ptb_dir, output_file):
    """Loops through all patient folders in ptb and extracts labels."""
    with open(output_file, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Folder Name", "Header File", "Reason for Admission"])
        
        for patient_folder in os.listdir(ptb_dir):
            patient_path = os.path.join(ptb_dir, patient_folder)
            if os.path.isdir(patient_path):
                for file in os.listdir(patient_path):
                    if file.endswith(".hea"):
                        header_path = os.path.join(patient_path, file)
                        reason = extract_label(header_path)
                        writer.writerow([patient_folder, os.path.splitext(file)[0], reason])

if __name__ == "__main__":
    ptb_directory = "ptb-diagnostic-ecg-database-1.0.0"  # Change this if needed
    output_csv = "new_labels.csv"
    process_ptb_directory(ptb_directory, output_csv)
    print(f"Labels extracted and saved to {output_csv}")