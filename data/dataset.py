import os
import cv2
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_img(img_path, img_width, img_height, augmentation=None, mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.shape != (img_height, img_width, 3):
        image = cv2.resize(image, (img_width, img_height))
    
    if augmentation:
        image = augmentation(image=image)
    image = image/255.

    return image-np.array(mean)/np.array(std)


def load_batch_img(img_dir, img_list, img_width, img_height, augmentation=None, mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225]):
    images = np.zeros((len(img_list), img_height, img_width, 3), dtype=np.float32)
    for i, img_name in enumerate(img_list):
        image = load_img(os.path.join(img_dir, img_name), img_width, img_height, augmentation=augmentation, mean=mean, std=std)
        images[i] = image
    return images


class Market1501:    
    le = LabelEncoder()
    
    def __init__(self, root_dir, test_size=None, random_state=None, shuffle=True):
        """
        root_dir: data directory
        test_size: float or int, default=None
        """
        self.root_dir = root_dir
        self.image_list, self.image_label = zip(*[(image_name, image_name[:4]) for image_name in os.listdir(root_dir) if image_name.endswith('.jpg')])
        self.image_label = self.le.fit_transform(self.image_label)
        if test_size is None:
            self.train_list = self.image_list
            self.train_label = self.image_label
            self.test_list = None
            self.test_label = None
        else:
            self.train_list, self.test_list, self.train_label, self.test_label = train_test_split(
                self.image_list,
                self.image_label,
                test_size=test_size,
                random_state=random_state,
                shuffle=shuffle
            )
        self.num_classes = len(self.le.classes_)
        self.anchor_pos_dic = {}
        for image_name in self.train_list:
            self.anchor_pos_dic.setdefault(image_name[:4], []).append(image_name)


    def get_train_loader(self, P=16, K=4, img_width=64, img_height=128, mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225], augmentation=None):
        """
        Sample triplet P*K (anchor, positive, negative)
        P: sample classes
        K: number of examples for each class
        augmentation: data augmentation
        """
        X_anchor = torch.zeros((P, K, 3, img_height, img_width), dtype=torch.float32)
        X_positive = torch.zeros((P, K, 3, img_height, img_width), dtype=torch.float32)
        X_negative = torch.zeros((P, K, 3, img_height, img_width), dtype=torch.float32)

        y_anchor = torch.zeros((P, K), dtype=torch.int64)
        y_positive = torch.zeros((P, K), dtype=torch.int64)
        y_negative = torch.zeros((P, K), dtype=torch.int64)
        
        while True:
            sampled_classes = np.random.choice(list(self.anchor_pos_dic.keys()), P, replace=False if P<=len(self.anchor_pos_dic) else True)
            for i, sampled_class in enumerate(sampled_classes):
                anchor_list = self.anchor_pos_dic[sampled_class]
                negative_list = list(set(self.train_list)-set(anchor_list))
                
                sampled_anchor_postive_list = np.random.choice(anchor_list, 2*K, replace=False if 2*K<=len(anchor_list) else True)
                sampled_anchor_list, sampled_positive_list = sampled_anchor_postive_list[:K], sampled_anchor_postive_list[K:]
                sampled_negative_list = np.random.choice(negative_list, K, replace=False if K<=len(negative_list) else True)
                
                X_anchor[i] = torch.from_numpy(load_batch_img(self.root_dir, sampled_anchor_list, img_width, img_height, augmentation=augmentation, mean=mean, std=std)).permute([0, 3, 1, 2])
                X_positive[i] = torch.from_numpy(load_batch_img(self.root_dir, sampled_positive_list, img_width, img_height, augmentation=augmentation, mean=mean, std=std)).permute([0, 3, 1, 2])
                X_negative[i] = torch.from_numpy(load_batch_img(self.root_dir, sampled_negative_list, img_width, img_height, augmentation=augmentation, mean=mean, std=std)).permute([0, 3, 1, 2])

                y_anchor[i] = torch.from_numpy(self.le.transform([sampled_class]*K))
                y_positive[i] = torch.from_numpy(self.le.transform([sampled_class]*K))
                y_negative[i] = torch.from_numpy(self.le.transform([image_name[:4] for image_name in sampled_negative_list]))
            
            #print(sampled_anchor_list)
            #print(sampled_positive_list)
            #print(sampled_negative_list)
            
            yield (X_anchor.flatten(end_dim=1), X_positive.flatten(end_dim=1), X_negative.flatten(end_dim=1)), (y_anchor.flatten(), y_positive.flatten(), y_negative.flatten())


    def get_test_loader(self, img_width=64, img_height=128, mean=[0.485, 0.456,0.406], std=[0.229, 0.224, 0.225], augmentation=None):
        if self.test_list is None:
            raise StopIteration("must set test_size not 0")

        X = torch.from_numpy(load_batch_img(self.root_dir, self.test_list, img_width, img_height, augmentation=augmentation, mean=mean, std=std)).permute([0, 3, 1, 2])
        y = torch.from_numpy(self.test_label)
        while True:
            yield X, y

if __name__ == "__main__":
    market1501 = Market1501("/students/julyedu_665963/market1501/bounding_box_train/", test_size=0.2)

    print(market1501.le.classes_)
    print(len(market1501.le.classes_))
    print(market1501.test_label)
    
    test_loader = market1501.get_test_loader()
    for X_test, y_test in test_loader:
        print(X_test[0])
        print(y_test)
        break

    train_loader = market1501.get_train_loader(P=16, K=4)
    for i, ((X_anchor, X_positive, X_negative), (y_anchor, y_positive, y_negative)) in enumerate(train_loader):
        print(X_anchor.shape, X_positive.shape, X_negative.shape)
        if i> 10:
            break
