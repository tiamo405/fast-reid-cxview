import torch
import cv2
import os

dir_train = "data/PMC_sup_nam/train"
dir_gallery = "data/PMC_sup_nam/gallery"
dir_query = "data/PMC_sup_nam/query"

for id in os.listdir(dir_query) :
    print(id)
    for fname in os.listdir(os.path.join(dir_query, id)):
        image = cv2.imread(os.path.join(dir_query, id, fname))
        cv2.imwrite(os.path.join("view/query", (id+'_'+ fname)), image)
        break

for id in os.listdir(dir_gallery) :
    print(id)
    for fname in os.listdir(os.path.join(dir_gallery, id)):
        image = cv2.imread(os.path.join(dir_gallery, id, fname))
        cv2.imwrite(os.path.join("view/gallery", (id+'_'+ fname)), image)
        break

for folder in os.listdir(dir_train) :
    for id in os.listdir(os.path.join(dir_train, folder)) :
        print(id)
        for fname in os.listdir(os.path.join(dir_train, folder, id)):
            image = cv2.imread(os.path.join(dir_train, folder, id, fname))
            cv2.imwrite(os.path.join("view/train", (folder+ '_'+id+'_'+ fname)), image)
            break
