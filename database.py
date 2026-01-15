import xml.etree.ElementTree as ET
import pandas as pd

# Path to ECG XML file
xml_path = "YOUR_FILE.xml"

# Parse XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Dictionary to store metadata
metadata = {}

# Iterate over all XML elements and search for known metadata fields
for elem in root.iter():
    tag = elem.tag.lower()

    # Subject / patient identifier
    if "patientid" in tag or "subjectid" in tag:
        metadata["subject_id"] = elem.text

    # Age
    if "age" in tag:
        metadata["age"] = elem.text

    # Sex / gender
    if "sex" in tag or "gender" in tag:
        metadata["sex"] = elem.text

    # Sampling rate (Hz)
    if "samplingrate" in tag or "samplefrequency" in tag:
        metadata["sampling_rate"] = float(elem.text)

# Convert metadata to a one-row DataFrame
metadata_df = pd.DataFrame([metadata])

metadata_df
