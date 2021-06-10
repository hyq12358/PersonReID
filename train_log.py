import config

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from criterion import TripletLoss, LabelSmoothingLoss
from metric import accuracy
from data.dataset import Market1501
from modeling.model import CNNModel
from utils.augment import seq
from utils.visualize import Visualizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


"""
Load Dataset market1501 and get dataloader
"""
market1501 = Market1501(root_dir=config.DATA_DIR, test_size=0.2, shuffle=True)
train_loader = market1501.get_train_loader(P=config.P, K=config.K, img_width=config.IMG_WIDTH, img_height=config.IMG_HEIGHT, augmentation=seq)
test_loader = market1501.get_test_loader(img_width=config.IMG_WIDTH, img_height=config.IMG_HEIGHT)

num_classes=market1501.num_classes


"""
Create  model
"""
model = CNNModel(num_classes=num_classes).to(device=device)
print(model)


"""
Set loss function
"""
if config.THETA_LABEL_SMOOTHING == 0:
    criterion_cls = nn.CrossEntropyLoss().to(device)
else:
    criterion_cls = LabelSmoothingLoss(num_classes, config.THETA_LABEL_SMOOTHING).to(device)

criterion_triplet = TripletLoss(alpha=.3)


"""
 Set optimizer(scheduler)
"""
optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

"""
Use tensorboard to visualize
"""
vis = Visualizer(config.LOGDIR)

"""
Set checkpoint
"""
ckt = "{}/person_reid.pth".format(config.CHECKPOINT_DIR)


def train(model, train_loader, steps, test_loader=None):
    """
    Train a model 

    Parameter:
    -----------------------------------------
    * model:
    * train_loader: 
    * test_loader:
    * num_training

    Return:
    -----------------------------------------
    """
    best_acc = None
    for step, ((X_anchor, X_positive, X_negative), (y_anchor, y_positive, y_negative)) in enumerate(train_loader):
        assert X_anchor.shape==X_positive.shape and X_anchor.shape==X_negative.shape and X_positive.shape==X_negative.shape, "anchor, positive, negative must have the same shape"
        n = X_anchor.size(0)

        X_anchor = X_anchor.to(device=device)
        X_positive = X_positive.to(device=device)
        X_negative = X_negative.to(device=device)

        y_anchor = y_anchor.to(device=device)
        y_positive = y_positive.to(device=device)
        y_negative = y_negative.to(device=device)

        X_train = torch.cat([X_anchor, X_positive, X_negative], dim=0)
        y_train = torch.cat([y_anchor, y_positive, y_negative], dim=0)

        #online hard example mining
        model.eval()
        with torch.no_grad():
            _, features = model(X_train)
        anchor_features, positive_features, negative_features = features.split(n, dim=0)
        similarity = torch.mm(anchor_features, negative_features.t())
        hard_negative_indices = torch.argmax(similarity, dim=-1)
        X_train[2*n:] = X_negative[hard_negative_indices]
        y_train[2*n:] = y_negative[hard_negative_indices]

        #train
        model.train()
        y_pred, features = model(X_train)
        optimizer.zero_grad()
        cls_loss = config.CLS_LOSS_WEIGHT*criterion_cls(y_pred, y_train)
        anchor_features, positive_features, negative_features = features.split(n, dim=0)
        tri_loss = config.TRI_LOSS_WEIGHT*criterion_triplet(anchor_features, positive_features, negative_features, normalize=False) 
        loss = cls_loss+tri_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (step+1)%10 == 0:
            acc = accuracy(y_pred.detach().cpu(), y_train.detach().cpu())
            loss = loss.detach().cpu().item()
            cls_loss = cls_loss.detach().cpu().item()
            tri_loss = tri_loss.detach().cpu().item()
            print("Train Step[{}], loss: {:.6f}, cls_loss: {:.6f}, tri_loss: {:.6f}, acc: {:.6f}".format(step+1, loss, cls_loss, tri_loss, acc))

            vis.log_scalars("Train", dict(train_loss=loss, train_cls_loss=cls_loss, train_tri_loss=tri_loss, train_acc=acc), global_step=step+1)
            
            if test_loader:
                X_test, y_test = next(test_loader)
                X_test = X_test.to(device=device)
                y_test = y_test.to(device=device)

                model.eval()
                with torch.no_grad():
                    y_pred, _ = model(X_test)
                model.train()
	        
                cls_loss = criterion_cls(y_pred, y_test).detach().cpu().item()
                acc = accuracy(y_pred.detach().cpu(), y_test.detach().cpu())

                print("Test after steps[{}], acc: {:.6f}, best_acc: {:.6f}".format(step+1, acc, best_acc if best_acc else acc))

                vis.log_scalars("Test", dict(test_cls_loss=cls_loss, test_acc=acc), global_step=step+1)

            if best_acc is None or acc>best_acc:
                torch.save(model.state_dict(), ckt, _use_new_zipfile_serialization=True if torch.__version__>="1.6" else False)
                best_acc = acc
        
        if (step+1) == config.STEPS:
            break
            

if __name__ == "__main__":
    if os.path.exists(ckt):
        model.load_state_dict(torch.load(ckt))
    train(model, train_loader, config.STEPS, test_loader)
