import argparse
from ast import arg
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
# from dataset_paths import DATASET_PATHS # 不需要这个了
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

# --- 新增引用 ---
from datasets import load_dataset, concatenate_datasets
import io
# ----------------

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader))) # 打印出是batch数量，batch很大
        for img, label in tqdm(loader):
            in_tens = img.cuda()

            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    print(len(y_true), y_true)
    print(len(y_pred), y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

    
    
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 

class RealFakeDataset(Dataset):
    def __init__(self, 
                 # 以下参数保留接口兼容，但 path 参数不再使用
                 real_path=None, 
                 fake_path=None, 
                 data_mode=None, 
                 max_sample=1000,
                 arch='imagenet',
                 jpeg_quality=None,
                 gaussian_sigma=None,
                 cache_dir=None):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        
        print(f"Loading dataset from Hugging Face... (Taking top {max_sample} per class, NO SHUFFLE)")
        
        # 1. 加载数据集
        ds_args = {"path": "OwensLab/CommunityForensics-Eval", "split": "CompEval"}
        if cache_dir:
            ds_args["cache_dir"] = cache_dir
            
        full_ds = load_dataset(**ds_args)

        # 2. 按顺序筛选真假样本 seed=42
        real_ds = full_ds.filter(lambda x: x['label'] == 0).shuffle(seed=42)
        fake_ds = full_ds.filter(lambda x: x['label'] == 1).shuffle(seed=42)

        # 3. 处理数量限制 (Max Sample) - 严格取前 N 个
        if max_sample is not None:
            # 取 min 防止越界，直接用 range 截取头部
            limit_real = min(len(real_ds), max_sample)
            limit_fake = min(len(fake_ds), max_sample)
            
            # 【核心修改】这里去掉了 shuffle，直接 select 前 limit 个
            real_ds = real_ds.select(range(limit_real))
            fake_ds = fake_ds.select(range(limit_fake))
            
            print(f"Selected top {limit_real} Real images and top {limit_fake} Fake images.")

        # 4. 合并 (保持顺序：先全是真，后全是假)
        self.dataset = concatenate_datasets([real_ds, fake_ds])
        
        # 兼容性字段
        self.total_list = range(len(self.dataset))

        # = = = = = = Transform = = = = = = = = = # 
        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 1. 获取图片
        try:
            image_bytes = item['image_data']
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            print(f"Error loading image at index {idx}: {e}")
            image = Image.new('RGB', (224, 224))

        # 2. 应用 Augmentation (JPEG Compression / Gaussian Blur)
        # 这一步必须在 Transform 转 Tensor 之前做
        if self.jpeg_quality is not None:
            image = png2jpg(image, self.jpeg_quality)
        elif self.gaussian_sigma is not None:
            image = gaussian_blur(image, self.gaussian_sigma)

        # 3. 获取标签
        label = item['label'] 
        
        # 4. 应用 Standard Transform (Resize, ToTensor, Normalize)
        if self.transform:
            image = self.transform(image)
            
        return image, label


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 这些 path 参数虽然不用了，但为了命令行兼容性保留
    parser.add_argument('--real_path', type=str, default=None, help='(Not used for HF dataset)')
    parser.add_argument('--fake_path', type=str, default=None, help='(Not used for HF dataset)')
    parser.add_argument('--data_mode', type=str, default=None, help='(Not used for HF dataset)')
    
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')
    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--result_folder', type=str, default='result', help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30.")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.")
    
    # 新增 cache_dir
    parser.add_argument('--cache_dir', type=str, default=None, help="Hugging Face Dataset Cache Dir")

    opt = parser.parse_args()
    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder)

    model = get_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location='cpu')
    model.fc.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda()

    # --- 修改部分：不再遍历 DATASET_PATHS，而是指定为 HF Dataset ---
    dataset_paths = [ 
        dict(real_path=None, fake_path=None, data_mode=None, key="HF_CommunityForensics") 
    ]
    # -----------------------------------------------------------

    for dataset_path in (dataset_paths):
        set_seed()
        print(f"deal with {dataset_path}")

        dataset = RealFakeDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'], 
                                    dataset_path['data_mode'], 
                                    opt.max_sample, 
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality, 
                                    gaussian_sigma=opt.gaussian_sigma,
                                    cache_dir=opt.cache_dir # 传入 cache_dir
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True)

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc1.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc1*100, 2))+'  '+str(round(f_acc1*100, 2))+'  '+str(round(acc1*100, 2))+'\n' )