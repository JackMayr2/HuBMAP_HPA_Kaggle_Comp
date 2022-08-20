import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()

lis1 = api.competitions_list(search='hubmap-organ-segmentation')
api.competition_download_files('hubmap-organ-segmentation')
kaggle.unzip
# open in finder, then unzip and folders will be in the correct place