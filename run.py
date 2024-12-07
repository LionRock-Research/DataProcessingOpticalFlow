import argparse
import time
import os
import torch
import numpy as np
import pandas as pd
import cv2
import json
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, DistributedSampler, ConcatDataset
from core.raft import RAFT
from core.utils.utils import InputPadder
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import transforms



DEVICE = 'cuda'

def load_s3_jsonl(s3_obj_list_file):
    s3_files_list = []
    with open(s3_obj_list_file, "r") as f:
        for line in f:
            json_object = json.loads(line.rstrip("\n"))
            s3_path, url = list(json_object.items())[0]
            s3_files_list.append((s3_path, url))

    return s3_files_list

def filter_unprocessed_files(s3_obj_list_file, parquet_file_path):
    s3_files_list = load_s3_jsonl(s3_obj_list_file)

    if not os.path.exists(parquet_file_path):
        return s3_files_list
    else:
        table = pq.read_table(parquet_file_path)
        processed_files = table.column("file_path").to_pylist()
        print(f'processed_files: {len(processed_files) / len(s3_files_list)}')
        unprocessed_files = [(file_path, url) for file_path, url in s3_files_list if file_path not in processed_files]
        return unprocessed_files
    
def get_data_type(data_list):
    """
    根据数据列表推断合适的Arrow数据类型。
    """
    if all(isinstance(x, bytes) for x in data_list):
        return pa.binary()
    return pa.string() if data_list else pa.null()

def update_existing_table(existing_table, new_table):
    """
    根据file_path列来更新已存在表中的同名行数据。
    """
    # 将 Arrow 表转换为 pandas DataFrame，方便进行数据操作（此处可根据实际性能需求考虑是否有更优转换方式）
    existing_df = existing_table.to_pandas()
    new_df = new_table.to_pandas()

    # 遍历新表中的每一行数据（根据file_path列来判断和更新）
    for index, row in new_df.iterrows():
        file_path_value = row['file_path']
        existing_index = existing_df[existing_df['file_path'] == file_path_value].index
        if len(existing_index) > 0:
            # 如果存在同名行，更新对应行的数据
            existing_index = existing_index[0]
            for col in existing_df.columns:
                if col!= 'file_path' and row[col] != None:
                    existing_df.at[existing_index, col] = row[col]
        else:
            # 如果不存在同名行，则添加该行到现有表对应的数据结构中（这里添加到pandas DataFrame）
            existing_df = pd.concat([existing_df, row.to_frame().T], ignore_index=True)

    # 将更新后的pandas DataFrame转换回Arrow表
    updated_table = pa.Table.from_pandas(existing_df)
    return updated_table

def write_lists_to_parquet(lists_data, file_path):
    """
    将包含不同元素的列表数据写入Parquet文件。

    参数:
    lists_data (list of lists): 包含多个列表的列表，每个子列表代表一次生成的数据
    file_path (str): Parquet文件的存储路径
    """
    field_names = ["file_path", "opt_mira", "opt_sora"]
    all_data = [[] for _ in range(len(field_names))]

    # 遍历每个生成的列表，将元素填充到对应的字段数据列表中
    for single_list in lists_data:
        filled_data = [None] * len(field_names)
        for index, element in enumerate(single_list):
            if index < len(field_names):
                if isinstance(element, list):
                    filled_data[index] = json.dumps(element)
                elif isinstance(element, bytes):
                    filled_data[index] = element
                else:
                    filled_data[index] = element

        for index, value in enumerate(filled_data):
            all_data[index].append(value)

    # 根据实际收集到的数据情况构建 Arrow 数组
    arrays = []
    for index, data_list in enumerate(all_data):
        data_type = get_data_type(data_list)
        arrays.append(pa.array(data_list, type=data_type))

    # 构建模式（Schema）
    fields = []
    for name, data_list in zip(field_names, all_data):
        field_type = get_data_type(data_list)
        fields.append(pa.field(name, field_type))
    schema = pa.schema(fields)

    # 构建 Arrow 表
    table = pa.Table.from_arrays(arrays, schema=schema)

    # 写入 Parquet 文件逻辑优化及异常处理添加
    try:
        if os.path.exists(file_path):
            # 如果文件已存在，尝试以追加模式写入
            existing_table = pq.read_table(file_path)
            if "file_path" in existing_table.column_names:
                # 处理同名 file_path 行更新逻辑
                new_table = update_existing_table(existing_table, table)
                pq.write_table(new_table, file_path, use_dictionary=True, compression='snappy')
            else:
                with pq.ParquetWriter(file_path, schema, use_dictionary=True, compression='snappy', append=True) as writer:
                    writer.write_table(table)
        else:
            # 如果文件不存在，直接写入新文件
            pq.write_table(table, file_path, use_dictionary=True, compression='snappy')
    except FileNotFoundError as e:
        print(f"文件不存在导致写入Parquet文件出错: {e}")
        raise
    except pq.ParquetException as e:
        print(f"Parquet文件格式相关错误: {e}")
        raise
    except Exception as e:
        print(f"写入Parquet文件时出现其他未知错误: {e}")
        raise

def dumb_fill_in(unprocessed_files_list, parquet_path):
    unprocessed_files = [item[0] for item in unprocessed_files_list]
    if len(unprocessed_files) > 0:
        if os.path.exists(parquet_path):
            table = pq.read_table(parquet_path)
            df = table.to_pandas()
        else:
            columns = ["file_path", "opt_mira", "opt_sora"]
            
            df = pd.DataFrame(columns=columns)

        new_data = pd.DataFrame({
        "file_path": unprocessed_files,
        "opt_mira": [None] * len(unprocessed_files),
        "opt_sora": [None] * len(unprocessed_files)
    })
        updated_df = pd.concat([df, new_data], ignore_index=True)

                # 将更新后的DataFrame转换为Arrow Table
        updated_table = pa.Table.from_pandas(updated_df)

        update_path = parquet_path.split(".parquet")[0] + "_update.parquet"

        pq.write_table(updated_table, update_path)
        print("填充完成")
    else:
        print("处理完成，无剩余")
    

class VideoDataset(Dataset):
    """
    Dataset for loading videos and preparing frame pairs for optical flow computation.
    """
    def __init__(self, video_file, i):
        self.idx = i
        self.video_file = video_file
        self.frame_pairs = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))  # 调整为统一大小
        ])
        self.last = None
        cap = cv2.VideoCapture(video_file)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.cap = None

    def __len__(self):
        """
        返回合理的数据集大小，确保为正整数，这里取帧对数量（总帧数减1）与0中的较大值
        """
        return max(0, self.frame_count - 1)

    def __getitem__(self, idx):
        """
        Returns a single frame pair (image1, image2).
        """
        # 直接使用在初始化时打开的视频捕获对象来读取帧
        if self.cap is None: # lazy init
            self.cap = cv2.VideoCapture(self.video_file)
        ret, frame = self.cap.read()
        if not ret:
            raise ValueError(f"无法读取索引为 {idx} 的帧")
        if self.last is None:
            self.last = frame
            ret, frame = self.cap.read()
            if not ret:
                raise ValueError(f"无法读取索引为 {idx} 的帧")
        padder = InputPadder(frame.shape)
        image1 = self._load_image(self.last)
        image2 = self._load_image(frame)
        image1, image2 = padder.pad(image1, image2)
        image1, image2 = self.transform(image1), self.transform(image2)
        self.last = frame
        if idx == self.frame_count - 1:
            self.cap.release()
        return image1, image2, torch.tensor(self.idx)

    def _load_image(self, img):
        """
        Convert a numpy.ndarray image to a torch tensor and move it to the target device.
        """
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        return img


def get_score(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    u = flo[:, :, 0]
    v = flo[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
        
    h, w = rad.shape
    rad_flat = rad.flatten()
    cut_index = int(h * w * 0.05)

    max_rad = np.mean(abs(np.sort(-rad_flat))[:cut_index])

    return max_rad.item()


def get_score2(flo):
    flo = flo[0].permute(1, 2, 0).cpu().numpy()
    abs_flo = np.abs(flo)
    mean_abs_flo = np.mean(abs_flo, axis=(0, 1, 2))
    return mean_abs_flo


def demo(args):
    # Load the RAFT model
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model = model.module
    model.to(DEVICE)
    model.eval()

    video_files = filter_unprocessed_files(args.s3_obj_list_file, args.target_dir)
        
    ids = {}
    datasets = []
    # Process each video
    i = 0
    start_time = time.time()
    for tuple in video_files:
        video_name, video_path = tuple
        # print(f"Preprocessing video: {video_path}", i, len(video_files))
        print(f"Preprocessing:", i, '/', len(video_files))
        ids[i] = video_name
        dataset = VideoDataset(video_path, i)
        datasets.append(dataset)
        i += 1
        if i >= 100:
            break
    
    dataset = ConcatDataset(datasets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"数据预处理用时: {execution_time} 秒")

    mira_scores = {}
    sora_scores = {}
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Loading"):
            image1_batch, image2_batch, idx = batch
            image1_batch = image1_batch.squeeze(1).to(DEVICE)
            image2_batch = image2_batch.squeeze(1).to(DEVICE)
                
            flow_low, flow_up = model(image1_batch, image2_batch, iters=20, test_mode=True)
            
            for i in range(flow_up.size(0)):
                mira_scores.setdefault(idx[i].item(), [])
                score = get_score(flow_up[i].unsqueeze(0))
                mira_scores[idx[i].item()].append(score)
                
                sora_scores.setdefault(idx[i].item(), [])
                score = get_score2(flow_up[i].unsqueeze(0))
                sora_scores[idx[i].item()].append(score)
                
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"推理用时: {execution_time} 秒")
            
    data = []
    for k in ids.keys():
        tmp = []
        tmp.append(ids[k])
        mira = str(float(np.mean(mira_scores[k])))
        sora = str(float(np.mean(sora_scores[k])))
        tmp.append(mira)
        tmp.append(sora)
        data.append(tmp)
    
    write_lists_to_parquet(data, args.target_dir)
    file_list = filter_unprocessed_files(args.s3_obj_list_file, args.target_dir)
    dumb_fill_in(file_list, args.target_dir)
    print('finished!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos and save detection results.")
    parser.add_argument("--s3_obj_list_file", required=True, help="Path to the directory containing videos.")
    parser.add_argument("--target_dir", required=True, help="Path to the output Parquet file.")
    parser.add_argument("--model", default="/root/model/RAFT/raft-things.pth", help="Path to RAFT model.")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument("--frame_sample_ratio", type=int, default=30, help="Frame sampling ratio.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size for model inference.")
    args = parser.parse_args()

    demo(args)
