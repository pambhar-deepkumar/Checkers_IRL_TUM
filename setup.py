import os

SUB_FOLDER = "8x8" # Change this to the folder name of the map you want to use

EVALUATION_RESULTS_FILENAME = f'./ignored_files/{SUB_FOLDER}/data/evaluation_results.json'
Q_TABLE_SAVE_BASENAME = f'./ignored_files/{SUB_FOLDER}/q_values/q_table'
# List of directories to create
directories = [
    os.path.dirname(EVALUATION_RESULTS_FILENAME),
    os.path.dirname(Q_TABLE_SAVE_BASENAME)
]

# Create each directory if it doesn't exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)
