from torch.utils.data import Dataset
from data_youtube import YouTubeDataset
from data_nuplan import nuPlanDataset


def get_dataset(data_dir):
    if "youtube" in data_dir:
        print("!!!Setting class to YouTubeDataset!!!")
        return YouTubeDataset
    elif "nuplan" in data_dir:
        print("!!!Setting class to nuPlanDataset!!!")
        return nuPlanDataset
    else:
        raise ValueError("Invalid data_dir: should contain either 'youtube' or 'nuplan'.")

class MultiSourceDataset(Dataset):
    def __init__(self, data_dir, video_size, fps, max_num_frames, **kwargs):
        # Get the appropriate dataset class based on `data_dir`
        # dataset_class = get_dataset(data_dir)
        # dataset_class.__init__(
        #     self,
        #     data_dir=data_dir,
        #     video_size=video_size,
        #     fps=fps,
        #     max_num_frames=max_num_frames,
        #     **kwargs
        # )

        # Initialize the appropriate dataset based on `data_dir` or other conditions
        if "youtube" in data_dir.lower():
            print("!!!Initializing YouTubeDataset!!!")
            self.dataset = YouTubeDataset(
                data_dir=data_dir,
                video_size=video_size,
                fps=fps,
                max_num_frames=max_num_frames,
                **kwargs
            )
        elif "navsim" in data_dir.lower():
            print("!!!Initializing nuPlanDataset!!!")
            self.dataset = nuPlanDataset(
                data_dir=data_dir,
                video_size=video_size,
                fps=fps,
                max_num_frames=max_num_frames,
                **kwargs
            )
        else:
            raise ValueError("Invalid data_dir: should contain either 'youtube' or 'nuplan'.")

    # # Delegate attribute and method calls to the dataset instance
    def __getattr__(self, name):
        # Only get attributes from the dataset instance if they aren’t found in MultiSourceDataset
        return getattr(self.dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    @classmethod
    def create_dataset_function(cls, path, args, **kwargs):
        return cls(data_dir=path, **kwargs)