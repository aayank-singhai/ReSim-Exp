from torch.utils.data import Dataset
from data_youtube import YouTubeDataset
from data_nuplan import nuPlanDataset
from data_carla import CarlaDataset


class MultiSourceDataset(Dataset):
    def __init__(self, data_dir, 
                 skip_frms_num=0, 
                 token_json=None, 
                 prefix_prompt="",
                 prefix_prompt_carla=None,
                 scene_tensor_json_folder=None,
                 with_human_drive_token=False,  # * True for nuPlan and YouTube, False for Carla
                 **kwargs):
        # Initialize the appropriate dataset based on `data_dir` or other conditions
        if "youtube" in data_dir.lower():
            print("!!!Initializing YouTubeDataset!!!")
            self.dataset = YouTubeDataset(
                data_dir=data_dir,
                prefix_prompt=prefix_prompt,
                with_human_drive_token=with_human_drive_token,
                **kwargs
            )
        elif "navsim" in data_dir.lower():
            print("!!!Initializing nuPlanDataset!!!")
            self.dataset = nuPlanDataset(
                data_dir=data_dir,
                prefix_prompt=prefix_prompt,

                # * Only nuplan uses them.
                skip_frms_num=skip_frms_num,   
                token_json=token_json,
                scene_tensor_json_folder=scene_tensor_json_folder,
                with_human_drive_token=with_human_drive_token,

                **kwargs
            )
        elif "carla" in data_dir.lower():
            print("!!!Initializing CarlaDataset!!!")
            self.dataset = CarlaDataset(
                data_dir=data_dir,
                prefix_prompt=prefix_prompt_carla or prefix_prompt,


                with_human_drive_token=False,  # * Always False for Carla
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