# TODO: Import necessary modules
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader  # use to load an image from disk

# from the DeepWeeds dataset for convenience
CLASS_NAMES = ['Chinee Apple',
               'Lantana',
               'Parkinsonia',
               'Parthenium',
               'Prickly Acacia',
               'Rubber Vine',
               'Siam Weed',
               'Snake Weed',
               'Negatives']

# TODO: Update DWDataset to load images and labels from the dataset
class DWDataset(Dataset):
    def __init__(self, ds_root_folder: str, labels_key: str="test_subset0", transforms=None):
        """
        Initialize the dataset with the given path and labels key.

        Args:
            ds_root_folder (str): Path to the dataset directory. Expects subdirectories `images` and `labels`.
            labels_key (str): Optional stem of the CSV file containing labels.
            transforms: Optional transforms to apply to images.
        """
        pass
    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

# TODO: Update display_images to plot a subset of images and display their labels
def display_images(dataset: DWDataset, num_images: int = 4):
    """
    Display random images from DeepWeed dataset with labels from CSV.

    Args:
        dataset: DWDataset instance
        num_images: Number of random images to display
    """
    pass
