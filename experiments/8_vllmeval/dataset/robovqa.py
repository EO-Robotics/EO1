# flake8: noqa
from ..smp import *
from .video_base import VideoBaseDataset
import pandas as pd
import os
from .utils.qbench_video import *

FAIL_MSG = "Failed to obtain answer via API."


class RoboVQA(VideoBaseDataset):
    MD5 = "49e6181b341c934d0b33ec78bdcc0a3d"

    FRAMES_TMPL_SYS = """
You will receive {} distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
You are a robot task planning assistant. Analyze these frames and provide a **short and accurate** answer following the exact format below.

**Rules:**

Answer "done" when goal completed, follow exact format, use specific actions, keep concise.

case 1. State Estimation, answer "yes" or "no"
    Example 1:
    human: image or video place the packet on the table Q: possible right now?
    gpt: yes

    Example 2:
    human: image or video put the glass into the trash Q: satisfied?
    gpt: yes

case 2. Task Planning, answer with a specific action OR "done" (if task completed), if containing "next N steps?", answer with a numbered list of N steps
    Example 1:
    human: image or video current goal is: Please remove the chips from the basket Q: immediate next step?
    gpt: place the packet on the table

    Example 2:
    human: image or video current goal is: Please grab a bowl and scoop mixed nuts into the bowl Q: immediate next step?
    gpt: done

    Example 3:
    current goal is: take the bag and cap on the desk and hang them on the coat rack. last 20 steps: 1- Move towards the table 2- Slightly move your arm down 3- Grab the carry bag 4- pick the carry bag 5- move to the right 6- move forward 7- hang the carry bag in the stand 8- Slightly move your arm back 9- Slightly move to the left 10- move towards the table 11- move your arm down 12- grab the cap 13- pick up the cap from the table 14- Move close to the coat rack 15- hang the cap on the coat rack Q: next 5 steps?
    1- Move to the left 2- done

    Example 4:
    human: image or video current goal is: Please place the seaweed from the tray. last 20 steps: 1- place the packet on the table 2- place the packet on the table 3- place the packet on the table 4- place the packet on the table 5- place the packet on the table Q: next 5 steps?
    gpt: 1- place the packet on the table 2- place the packet on the table 3- place the packet on the table 4- place the packet on the table 5- place the packet on the table

    Example 5:
    human: image or video Q: what is likely to happen next?
    gpt:  put book on the table


case 3. Event Understanding, answer with a specific action
    Example 1:
    human: image or video Q: what just happened?
    gpt: pick up the sanitizer

And More Examples...

Please analyze the frames and answer accordingly.
"""

    FRAMES_TMPL_SYS_4VIDEO_LLM = """
You will receive several distinct frames that have been uniformly sampled from a video sequence, arranged in the same temporal order as they appear in the video.
Please analyze these frames and provide a **short and accurate** answer answer following the exact format below.
**Rules:**

Answer "done" when goal completed, follow exact format, use specific actions, keep concise.

case 1. State Estimation, answer "yes" or "no"
    Example 1:
    human: image or video place the packet on the table Q: possible right now?
    gpt: yes

    Example 2:
    human: image or video put the glass into the trash Q: satisfied?
    gpt: yes

case 2. Task Planning, answer with a specific action OR "done" (if task completed), if containing "next N steps?", answer with a numbered list of N steps
    Example 1:
    human: image or video current goal is: Please remove the chips from the basket Q: immediate next step?
    gpt: place the packet on the table

    Example 2:
    human: image or video current goal is: Please grab a bowl and scoop mixed nuts into the bowl Q: immediate next step?
    gpt: done

    Example 3:
    current goal is: take the bag and cap on the desk and hang them on the coat rack. last 20 steps: 1- Move towards the table 2- Slightly move your arm down 3- Grab the carry bag 4- pick the carry bag 5- move to the right 6- move forward 7- hang the carry bag in the stand 8- Slightly move your arm back 9- Slightly move to the left 10- move towards the table 11- move your arm down 12- grab the cap 13- pick up the cap from the table 14- Move close to the coat rack 15- hang the cap on the coat rack Q: next 5 steps?
    1- Move to the left 2- done

    Example 4:
    human: image or video current goal is: Please place the seaweed from the tray. last 20 steps: 1- place the packet on the table 2- place the packet on the table 3- place the packet on the table 4- place the packet on the table 5- place the packet on the table Q: next 5 steps?
    gpt: 1- place the packet on the table 2- place the packet on the table 3- place the packet on the table 4- place the packet on the table 5- place the packet on the table

    Example 5:
    human: image or video Q: what is likely to happen next?
    gpt:  put book on the table

case 3. Event Understanding, answer with a specific action
    Example 1:
    human: image or video Q: what just happened?
    gpt: pick up the sanitizer

And More Examples...

Please analyze the frames and answer accordingly.
"""

    TYPE = "Video-VQA"

    def __init__(self, dataset="RoboVQA", root="val", data_file="val.jsonl", nframe=0, fps=-1, pack=False):
        self.dataset_name = dataset
        self.data_root = root
        self.data_file = data_file

        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, "images", dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = "frame-{}-of-{}.jpg"
        self.frame_tmpl_fps = "frame-{}-of-{}-{}fps.jpg"

        data = {"id": [], "question": [], "answer": [], "video": []}

        # load jsonl file
        with open(self.data_file) as f:
            for line in f:
                line = json.loads(line)
                for i in range(len(line["conversations"]) // 2):
                    question = line["conversations"][i * 2]["value"].replace("<video>", "")
                    answer = line["conversations"][i * 2 + 1]["value"]
                    data["id"].append(line["id"])
                    data["question"].append(question)
                    data["answer"].append(answer)
                    data["video"].append(line["video"])
        self.data = pd.DataFrame(data)

        if "index" not in self.data:
            self.data["index"] = np.arange(len(self.data))

        assert "question" in self.data and "video" in self.data
        videos = list(set(self.data["video"]))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError("fps and nframe should not be set at the same time")
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError("fps and nframe should be set at least one valid value")

    @classmethod
    def supported_datasets(cls):
        return ["RoboVQA"]

    def prepare_dataset(self, dataset_name="RoboVQA"):
        return dict(root=self.data_root, data_file=self.data_file)

    def save_video_frames(self, line):
        video = line["video"]
        vid_path = os.path.normpath(os.path.join(self.data_root, line["video"]))

        import torch
        import torchvision.io as tvio

        # Read video using torchvision
        vid, audio, info = tvio.read_video(vid_path, pts_unit="sec")

        # Get video information
        video_info = {
            "fps": info["video_fps"],
            "n_frames": vid.shape[0],
        }

        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info["n_frames"] / video_info["fps"]
            required_frames = int(total_duration * self.fps)
            step_size = video_info["fps"] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            # Convert tensor frames to PIL Images
            # vid shape: (T, H, W, C) - convert to PIL Images
            images = []
            for i in indices:
                if i < len(vid):
                    # Convert tensor to numpy array and then to PIL Image
                    frame = vid[i].numpy()  # Shape: (H, W, C)
                    # Ensure values are in [0, 255] range for uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    pil_image = Image.fromarray(frame)
                    images.append(pil_image)

            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        video_path = os.path.normpath(os.path.join(self.data_root, line["video"]))
        if video_llm:
            message = [dict(type="text", value=self.FRAMES_TMPL_SYS_4VIDEO_LLM)]
            message.append(dict(type="video", value=video_path))
            message.append(dict(type="text", value=line["question"]))
        else:
            img_frame_paths = self.save_video_into_images(line)
            message = [dict(type="text", value=self.FRAMES_TMPL_SYS.format(len(img_frame_paths)))]
            for im in img_frame_paths:
                message.append(dict(type="image", value=im))
            message.append(dict(type="text", value=line["question"]))
        return message

    @staticmethod
    def extract_characters_regex(s, choices=["(A)", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)"]):
        if type(s) is dict:
            s = ""
        s = s.strip()

        answer_prefixes = [
            "The best answer is",
            "The correct answer is",
            "The answer is",
            "The answer",
            "The best option isThe correct option is",
            "Best answer:Best option:",
            "possible right now?",
            "satisfied?",
            "immediate next step?",
            "next 5 steps?",
            "what is likely to happen next?",
            "what just happened?",
            "gpt:",
            "A:",
            "**",
        ]

        for answer_prefix in answer_prefixes:
            s = s.replace(answer_prefix, "")

        return s

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        import sacrebleu
        from ..smp import load

        data = load(eval_file)
        data["prediction"] = [str(x) if x is not None else " " for x in data["prediction"]]

        micro_metric = {"BLEU": 0, "total": 0}
        for index, it in data.iterrows():
            micro_metric["total"] += 1

            answer = "A: " + str(it["answer"])
            prediction = "A: " + self.extract_characters_regex(str(it["prediction"]))

            bleu = sacrebleu.sentence_bleu(answer, [prediction])
            micro_metric["BLEU"] += bleu.score

        accuracy_dict = {"BLEU": [micro_metric["BLEU"] / micro_metric["total"]]}
        result_df = pd.DataFrame(accuracy_dict)
        suffix = eval_file.split(".")[-1]
        score_file = eval_file.replace(f".{suffix}", "_acc.csv")
        dump(result_df, score_file)
        return result_df
