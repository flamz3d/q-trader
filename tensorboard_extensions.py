import git

def tbx_git_changeset():
		repo = git.Repo(search_parent_directories=True)
		return repo.head.object.hexsha