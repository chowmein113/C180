import json
from align_image_code import align_images
import skimage.io as skio
import os.path as osp
import os
import matplotlib.pyplot as plt
import re
import cv2
import numpy as np
import json
#globals
ALIGN_IMG = False
ASF_TO_JSON = True
def get_correspondence_pts_from_JSON(json_path: str) -> dict:
    data = {}
    with open(json_path) as f:
        data = json.load(f)
    return data

def load_pts(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    
    num_pts = int(lines[1].split(":")[1].strip())

    
    points = []
    for i in range(3, 3 + num_pts):
        x, y = map(float, lines[i].split())
        points.append((x, y))

    return np.array(points)
def load_asf_pts(filename, img_shape):
    height, width = img_shape[:2]
    corners = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    points = [] + corners
    
    for line in lines[16:]:
        if line.startswith('#') or len(line.strip()) == 0:
            continue
        
        parts = line.split()
        # Assuming the x and y coordinates are the 3rd and 4th items on each line
        if len(parts) > 5: 
            x, y = float(parts[2]), float(parts[3])
            points.append((int(np.round(x * width)), int(np.round(y * height))))
    # points.extend(corners)
    return points
def load_asf_dataset(img_dir_path: str, pts_dir_path: str):
    dataset = {}
    for fn in sorted(os.listdir(img_dir_path)):
        if re.search(r'(1)(m).(jpg)', fn):
            img = skio.imread(osp.join(img_dir_path, fn)) / 255
            pts = load_asf_pts(osp.join(pts_dir_path, fn[:-3] + 'asf'), img.shape)
            dataset[fn] = {'img': img,
                        "pts": pts}
    return dataset
def load_dataset(img_dir_path: str, pts_dir_path: str):
    dataset = {}
    for fn in sorted(os.listdir(img_dir_path)):
        if re.search(r'.(jpg)', fn):
            img = cv2.imread(osp.join(img_dir_path, fn))
            pts = load_pts(osp.join(pts_dir_path, fn[:-3] + 'pts'))
            dataset[fn] = {'img': img,
                        "pts": pts}
    return dataset
            

def main():
    img_folder = osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), "images")
    if ALIGN_IMG:
        
        name1 = input("Name of img1:\n")
        img1 = osp.join(img_folder, name1)
        name2 = input("Name of img2:\n")
        img2 = osp.join(img_folder, name2)
        im1 = skio.imread(img1) / 255
        im2 = skio.imread(img2) / 255
        aligned_imgs = align_images(im1, im2)
        
        plt.imsave(osp.join(img_folder, f"aligned_{name1}"), aligned_imgs[0])
        plt.imsave(osp.join(img_folder, f"aligned_{name2}"), aligned_imgs[1])
    if ASF_TO_JSON:
        img_path = osp.join(img_folder, "imm_face_db", "01-1m.jpg")
        json_path = osp.join(img_folder, "export.json")
        imname="01-1m"
        im = skio.imread(img_path)
        pts = load_asf_pts(img_path[:-3] + "asf", im.shape)
        dic = {
            "im1_name": imname,
            "im2_name":imname,
            "im1Points":pts,
            "im2Points": pts}
        with open(json_path, 'w') as f:
            json.dump(dic, f)
    

if __name__ == "__main__":
    main()