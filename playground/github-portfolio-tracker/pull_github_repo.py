from github import Github
from github import Auth

def get_folder_contents(repo, folder_name):
    contents = repo.get_contents(folder_name)

    for content in contents:
        print(f"{content.type} {content.path}")
    return contents

tracked_folders =["data-engineering", "playground"]

token = ''
auth = Auth.Token(token)
g = Github(auth=auth)

repo = g.get_repo("MateenMullah/engineering-portfolio")
#print(repo.get_contents(""))

for folder in tracked_folders:
    print(f"\nInspecting: {folder}")
    get_folder_contents(repo, folder)