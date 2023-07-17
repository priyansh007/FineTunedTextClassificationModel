from huggingface_hub import notebook_login, HfApi
import os


#To-Do: add your API token
api_token = ""
api = HfApi(token=api_token)
api.create_repo(repo_id="tweet_sentiments_40k")

repo_id="pzalavad/tweet_sentiments_40k"
folder_path = './model'
files = os.listdir(folder_path)
for file_name in files:
  api.upload_file(
      path_or_fileobj=os.path.join(folder_path, file_name),
      path_in_repo=file_name,
      repo_id=repo_id,
      repo_type="model",
  )