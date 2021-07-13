import config

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from metric import top1_accuracy
from modeling.model import CNNModel
from data.dataset import load_batch_img

device = torch.device("cpu")
num_classes = 751
ckt = "{}/person_reid.pth".format(config.CHECKPOINT_DIR)

if __name__ == "__main__":
    model = CNNModel(backbone='resnet50', num_classes=num_classes)

    query_list, query_label = zip(*[(image_name, int(image_name[:4])) for image_name in os.listdir(config.QUERY_DIR) if image_name.endswith(".jpg")])
    gallary_list, gallary_label = zip(*[(image_name, int(image_name[:4])) for image_name in os.listdir(config.GALLARY_DIR) if image_name.endswith(".jpg") and image_name not in query_list])

    try:
        model.load_state_dict(torch.load(ckt))
    except Exception as e:
        print("load model checkpoint %s failed: %s" %(ckt, e))
        exit(-1)

    print("load model checkpoint %s success!" %(ckt))

    model = model.to(device)

    X_query = torch.from_numpy(load_batch_img(config.QUERY_DIR, query_list, config.IMG_WIDTH, config.IMG_HEIGHT)).permute([0, 3, 1, 2]).to(device)
    X_gallary = torch.from_numpy(load_batch_img(config.GALLARY_DIR, gallary_list, config.IMG_WIDTH, config.IMG_HEIGHT)).permute([0, 3, 1, 2]).to(device)

    model.eval()
    with torch.no_grad():
        _, query_features = model(X_query)
        _, gallary_features = model(X_gallary)

    query_label = torch.LongTensor(query_label)
    gallary_label = torch.LongTensor(gallary_label)

    top1 = top1_accuracy(query_label, query_features, gallary_label, gallary_features, normalize=False)
    print("top1_accuracy: {:.6f}".format(top1))
    
    
