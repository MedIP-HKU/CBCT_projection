from huggingface_hub import login, hf_hub_download, snapshot_download, list_repo_files
login()
repo_id = 'hku-mip/cbct_projection'
repo_type = 'dataset'
download_path = '../../data'

# Select a subdirectory to download
folder = "xvi/brain/HN_P001/"
files = list_repo_files(repo_id, repo_type=repo_type)
target_files = [f for f in files if f.startswith(folder)]
for f in target_files:
    hf_hub_download(repo_id=repo_id,
                    filename=f,
                    repo_type=repo_type,
                    local_dir=download_path)

