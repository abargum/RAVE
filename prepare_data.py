import os
import argparse
import shutil
import pandas as pd

def organize_clips(tsv_file, clips_folder, output_folder):
    # Read the TSV file into a pandas DataFrame
    df = pd.read_csv(tsv_file, sep='\t')

    # Create a dictionary to store the mapping of ids to corresponding folder paths
    id_folder_mapping = {}

    # Iterate through the DataFrame and create folders for each id
    for index, row in df.iterrows():
        clip_id = str(row['client_id'])[:5]  # Take the first 5 characters of the id
        folder_path = os.path.join(output_folder, clip_id)

        # Create the folder if it doesn't exist
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Store the mapping of id to folder path
        id_folder_mapping[clip_id] = folder_path

    # Iterate through the audio clips and move them to the corresponding folders
    for index, row in df.iterrows():
        clip_id = str(row['client_id'])[:5]  # Take the first 5 characters of the id
        clip_pathname = row['path']

        # Extract the filename from the pathname
        clip_filename = os.path.basename(clip_pathname)
        print(clip_filename)

        # Create the new file path with .wav extension
        new_file_path = os.path.join(id_folder_mapping[clip_id], clip_filename.replace('.mp3', '.wav'))

        # Convert and move the file
        os.system(f'ffmpeg -i {os.path.join(clips_folder, clip_filename)} {new_file_path}')
        print(f'Moved and converted {clip_filename} to {new_file_path}')

        # Remove the original MP3 file
        os.remove(os.path.join(clips_folder, clip_filename))
        print(f'Removed original MP3 file: {clip_filename}')

def main():
    parser = argparse.ArgumentParser(description='Organize audio clips based on a TSV file.')
    parser.add_argument('--tsv_path', required=True, help='Path to the TSV file.')
    parser.add_argument('--clips_path', required=True, help='Path to the clips folder.')
    parser.add_argument('--output_path', default='output', help='Path to the output folder.')
    args = parser.parse_args()

    tsv_file_path = args.tsv_path
    clips_folder_path = args.clips_path
    output_folder_path = args.output_path

    organize_clips(tsv_file_path, clips_folder_path, output_folder_path)

if __name__ == "__main__":
    main()
