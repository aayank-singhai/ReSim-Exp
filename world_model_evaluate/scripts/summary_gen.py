import json
import os
import argparse

from tqdm import tqdm

ACCEPTABLE_IMG_FILES = [".jpg", ".jpeg", ".png", ".bmp"]
UP_COLLECT = None #["youtube_val_i2vgen-xl", "youtube_val_compare_ours_all"]
VID_COLLECT = None
COND_FRAME = "youtube_val_i2vgen-xl"


def summary_folder(folder_path, num_segments=1, max_counts=-1, mode="pseudo-video", seg=False, name="summary", segname="_summaries", has_no_up=False):
    """
    Generate a summary file for any given dataset folder.

    ARGS:
    > folder_path: str, the path to the folder containing the dataset.
    > num_segments: int, the number of segments to divide the dataset into.     
                    Will generate [num_segments] summary files.
    > max_counts: int, the maximum number of frames to use for each dataset segment.
    > mode: str, the mode of the dataset. "pseudo-video" or "video-file".
    > seg: bool, whether to split the summary file into multiple segments.
    """

    summary_folder_pseudo_video = listize_folder_pseudo_video

    print("Start collecting folder info...")
    summary_path = os.path.join(folder_path, "{}_all.json".format(name))
    if seg and not os.path.exists(os.path.join(folder_path, segname)):
        os.mkdir(os.path.join(folder_path, segname))
    exists = False
    for file in os.listdir(folder_path):
        if "_all.json" in file:
            print("Found possible summary file: {}".format(file))
            if input("Use this file as overall summary? (Y/N): ").lower() == "y":
                summary_path = os.path.join(folder_path, file)
                exists = True
                break
    if exists:
        print("Summary file already exists. Loading...")
        with open(summary_path, "r") as f:
            summary = json.load(f)
    elif mode == "pseudo-video":
        summary = summary_folder_pseudo_video(folder_path, has_no_up)
    else:
        raise ValueError("Unknown dataset mode [{}]".format(mode))
    
    if not exists:
        with open(summary_path, "w") as f:
            output = dict()
            output["freq"] = "<TODO>"
            output["gen_startid"] = "<TODO>"
            output["path"] = summary
            json.dump(output, f, indent=4)
    
    if seg:
        split(folder_path, summary, num_segments, max_counts, segname=segname)


def listize_folder_pseudo_video(folder_path, has_no_up=False):
    """
    Generate a summary file for a folder containing pseudo-videos.

    ARGS:
    > folder_path: str, the path to the folder containing pseudo-videos.
    
    RETURNS:
    > summary: dict, the summary file.
    """

    if has_no_up:
        up_list = [""]
    else:
        up_list = tqdm(os.listdir(folder_path))

    full_summary = []
    for up in up_list:
        # up = "/".join([up, "NuScenes", "virtual", "images"])
        up = "/".join([up, "virtual", "images"])
        up_dir = os.path.join(folder_path, up)
        if not os.path.isdir(up_dir):
            continue
        if (UP_COLLECT is not None) and (len(UP_COLLECT)>0) and (up not in UP_COLLECT):
            continue

        summary = []
        for video_id in os.listdir(up_dir):
            # video_id = "/".join([video_id, "obs"])
            video_dir = os.path.join(up_dir, video_id)
            if not os.path.isdir(video_dir):
                continue
            if (VID_COLLECT is not None) and (len(VID_COLLECT)>0) and (video_id not in VID_COLLECT):
                continue

            file_list = []
            cond = None
            for file in os.listdir(video_dir):
                is_image = False
                for frmt in ACCEPTABLE_IMG_FILES:
                    if file.endswith(frmt):
                        is_image = True
                        break
                if is_image:
                    if "cond" in file:
                        cond = os.path.join(up, video_id, file)
                    else:
                        file_list.append(os.path.join(up, video_id, file))

            file_list.sort()
            if cond is not None:
                file_list = [cond] + file_list
            summary.append(file_list)

        # if not has_no_up:
        # with open(os.path.join(folder_path, "{}_{}.json".format(up, "all")), "w") as f:
        #     output = dict()
        #     output["freq"] = "<TODO>"
        #     output["gen_startid"] = "<TODO>"
        #     output["path"] = summary
        #     json.dump(output, f, indent=4)

        full_summary.extend(summary)

    full_summary.sort()
    return full_summary


def split(folder_path, summary, num_segments=1, max_counts=-1, segname="_summaries"):
    """
    Split a summary file into multiple segments.

    ARGS:
    > summary: dict, the summary file to split.
    > num_segments: int, the number of segments to divide the dataset into. 
                    Will generate [num_segments] summary files.
    > max_counts: int, the maximum number of frames to use for each dataset segment.
    """

    if max_counts == -1:
        total = summary["meta"]["total_frames"] if "total_frames" in summary["meta"].keys() else summary["meta"]["num_videos"]
        max_counts = total // num_segments

    print("current settings: max_counts: {} num_segments: {}".format(max_counts, num_segments))

    while True:
        segment_summaries = []
        current_videolist = []
        count = 0
        true_max_count = 0
        for video in summary["videos"]:
            if true_max_count < count:
                true_max_count = count
            if "num_frames" in video.keys():
                add = video["num_frames"]
            else:
                add = 1
            # print(count, end=" ")

            if count+add >= max_counts:
                segment_summaries.append({
                    "meta": {
                        "frame_rate": summary["meta"]["frame_rate"],
                        "total_counts": count,
                    },
                    "videos": current_videolist
                })
                current_videolist = []
                count = 0
                # print()

            count += add
            current_videolist.append(video)

        if len(current_videolist) > 0:
            segment_summaries.append({
                "meta": {
                    "frame_rate": summary["meta"]["frame_rate"],
                    "total_counts": count,
                },
                "videos": current_videolist
            })

        print("Num of segments: {} Max counts: {}".format(len(segment_summaries), true_max_count))
        print("Are you sure to use this segment?")
        answer = input("Y/N/Q(uit): ").lower()
        if answer == "y":
            idx_width = len(str(len(segment_summaries)))
            for i in range(len(segment_summaries)):
                with open(os.path.join(folder_path, segname, "summary_{}.json".format(str(i).zfill(idx_width))), "w") as f:
                    json.dump(segment_summaries[i], f, indent=4)
            break
        elif answer == "q":
            break
        else:
            max_counts = int(input("New max_counts: "))
            num_segments = int(input("New num_segments: "))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="/cpfs01/shared/opendrivelab/opendrivelab_hdd/GenAD_Proj/SDXL_sampling/nusc_inter_resume_new/", help="The path to the folder containing the dataset.")
    parser.add_argument("--num_segments", type=int, default=16, help="The number of segments to divide the dataset into. Will generate [num_segments] summary files.")
    parser.add_argument("--max_counts", type=int, default=1000000, help="The maximum number of frames to use for each dataset segment.")
    parser.add_argument("--mode", type=str, default="pseudo-video", help="The mode of the dataset. \"pseudo-video\" or \"video-file\".")
    parser.add_argument("--seg", action="store_true", help="Whether to split the summary file into multiple segments.")
    parser.add_argument("--name", type=str, default="summary", help="The name of the dataset.")
    parser.add_argument("--segname", type=str, default="_summaries", help="The name of the folder to store the summary files.")
    parser.add_argument("--without_subfolders", action="store_true")
    args = parser.parse_args()

    args.without_subfolders = False

    summary_folder(args.folder_path, args.num_segments, args.max_counts, args.mode, args.seg,
                   name=args.name, segname=args.segname, has_no_up=args.without_subfolders)