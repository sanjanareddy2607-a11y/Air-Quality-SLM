import shutil
import os

os.makedirs('HF_Air_Quality_Dataset', exist_ok=True)
shutil.copy('preprocess_air_quality.py', 'HF_Air_Quality_Dataset/preprocess_air_quality.py')
shutil.copy('validate_dataset.py', 'HF_Air_Quality_Dataset/validate_dataset.py')

archive_name = 'HF_Air_Quality_Dataset'
shutil.make_archive(archive_name, 'zip', 'HF_Air_Quality_Dataset')
zip_size = os.path.getsize(archive_name + '.zip') / (1024 * 1024)
print(f"Archive created: {archive_name}.zip ({zip_size:.2f} MB)")

# List files in HF_Air_Quality_Dataset
files = os.listdir('HF_Air_Quality_Dataset')
print("Files in HF_Air_Quality_Dataset:")
for f in files:
    fp = os.path.join('HF_Air_Quality_Dataset', f)
    print(f" - {f:25} ({os.path.getsize(fp):,} bytes)")
