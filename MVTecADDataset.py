from PIL import Image
import sys, os, urllib.request, tarfile, cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import urllib.request
import zipfile
import random

import torch
import torch.utils.data as data
from torchvision import transforms

"""
test MVTec AD dataset
ref: https://github.com/shinmura0/AutoEncoder_vs_MetricLearning/blob/master/L2_vs_Autoencoder.ipynb
"""

#########################
# download data
###
def download_AD_dataset(source_path):
  download_dir = "./ad"
  save_path = "data/"

  if not os.path.exists(download_dir):
      os.mkdir(download_dir)

  # download file
  def _progress(count, block_size, total_size):
      sys.stdout.write('\rDownloading %s %.2f%%' % (source_path,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
  new_file_name = os.path.basename(source_path)
  dest_path = os.path.join(download_dir, new_file_name)
  if os.path.exists(dest_path):
      print(new_file_name+" is downloaded already")
  else:
      urllib.request.urlretrieve(source_path, filename=dest_path, reporthook=_progress)
  # untar
  with tarfile.open(dest_path, "r:xz") as tar:
      tar.extractall(save_path)


#########################
# MVTec AD dataset
###
class ADDataset(data.Dataset):
    def makeFileList(self,dir_path,data_name):
      retList = []
      TopDataFolder = dir_path
      DataFolder = TopDataFolder + "/"  +data_name
      subFolders = [str(os.path.basename(p)) for p in Path(DataFolder).glob("*")]
      for name in subFolders:
        #print("==",name,":")
        imagefolder = DataFolder+"/"  +name
        for f in Path(imagefolder).glob("*"):
            retList.append([str(os.path.basename(f)),name])
      #     
      #retList =retList +fileList
      #
      return retList

    def __init__(self, dir_path, data_name,input_size):
        super().__init__()
        
        self.dir_path = dir_path
        self.data_name = data_name
        self.input_size = input_size
        self.dataLists = self.makeFileList(dir_path,data_name)

        
    def __len__(self):
        return len(self.dataLists)
    
    def __getitem__(self, index):
        file,lable = self.dataLists[index]
        #print("file: ",file,"lable: ",lable)
        image_file = self.dir_path + "/" + self.data_name + "/"+lable+ "/"+file
        #print("to open: ",image_file)
        # open image file
        image = Image.open(image_file)
        image = image.resize(self.input_size)
        image = np.array(image)
        
        if(len(image.shape)<3):
            image = np.stack((image,)*3, axis=-1)
         
        image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image)
        
     
        return image,lable