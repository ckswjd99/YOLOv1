import torch
import torchvision.transforms as transforms
import torch.optim as optim
import sys
from torch.utils.data import DataLoader
from dataset import VOCDataset
from model import Yolov1
from utils import (
    get_bbox,
    non_max_suppression,
    load_checkpoint,
)

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
IMG_DIR = "data/data/images"
LABEL_DIR = "data/data/labels"
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
NUM_WORKERS = 2
PIN_MEMORY = True

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])

    test_dataset = VOCDataset(
        "data/test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    if len(sys.argv) > 1 and sys.argv[1] == '-l':
        load_checkpoint(torch.load(sys.argv[2]), model, optimizer)
    
    pred_boxes, target_boxes = get_bbox(test_loader, model, iou_threshold=0.5, threshold=0.4)

    print(len(pred_boxes), pred_boxes[0], len(target_boxes), target_boxes[0])

if __name__ == "__main__":
    main()