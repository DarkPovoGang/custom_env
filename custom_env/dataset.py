import numpy as np
import torch 
import clip 
import pandas as pd
import os
import gdown
import tarfile
from PIL import Image #, ImageDraw
import json
from torchvision import transforms
from torch.utils.data import Dataset


class RefCOCOg(Dataset):
    FILE_ID = "1wyyksgdLwnRMC9pQ-vjJnNUn47nWhyMD"
    ARCHIVE_NAME = "refcocog.tar.gz"
    NAME = "refcocog"
    ANNOTATIONS = "annotations/refs(umd).p"
    JSON = "annotations/instances.json"
    IMAGES = "images"
    IMAGE_NAME = "COCO_train2014_{}.jpg"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self._check_dataset()
        self.split = split
        self._filter_annotation(
            os.path.join(self.data_dir, self.NAME, self.ANNOTATIONS)
        )
        self._load_json()
        self.transform = transform
        self.model, self.preprocess = clip.load("RN50")

    def _check_dataset(self):
        if not os.path.exists(os.path.join(self.data_dir, self.ARCHIVE_NAME)):
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            print("Downloading dataset...")
            gdown.download(id=self.FILE_ID)
        if not os.path.exists(os.path.join(self.data_dir, self.NAME)):
            print("Extracting dataset...")
            with tarfile.open(
                os.path.join(self.data_dir, self.ARCHIVE_NAME), "r:gz"
            ) as tar:
                tar.extractall(path=self.data_dir)
        else:
            print("Dataset already extracted")

    def _load_json(self):
        with open(os.path.join(self.data_dir, self.NAME, self.JSON)) as f:
            self.json = json.load(f)
        self.json = pd.DataFrame(self.json["annotations"])

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, idx):
        # get line by index
        raw = self.annotation.iloc[idx]
        # get image
        image = self._get_image(raw)
        # get sentences
        sentences = self._get_sentences(raw)
        # get bbox
        bboxes = self._get_bboxes(raw)

        # return self._get_vector(image, sentences, bboxes)
        return self._get_vector(image, sentences) , bboxes

    def _get_image(self, raw):
        # get image_id
        image_id = raw["image_id"]
        # pad image_id to 12 digits
        image_id = str(image_id).zfill(12)
        # convert image to tensor
        image = Image.open(
            os.path.join(
                self.data_dir, self.NAME, self.IMAGES, self.IMAGE_NAME.format(image_id)
            )
        )
        return image

    def _get_sentences(self, raw):
        # get sentences
        sentences = raw["sentences"]
        # get raw sentences
        sentences = [sentence["raw"] for sentence in sentences]
        return sentences

    def _get_bboxes(self, raw):
        # get ref_id
        id = raw["ann_id"]
        bboxes = self.json[self.json["id"] == id]["bbox"].values[0]
        return bboxes

    def _filter_annotation(self, path):
        self.annotation = pd.read_pickle(path)
        self.annotation = pd.DataFrame(self.annotation)
        self.annotation = self.annotation[self.annotation["split"] == self.split]

    def _get_vector(self, image, sentences):
        image = self.preprocess(image).unsqueeze(0).to(RefCOCOg.DEVICE)
        text = clip.tokenize(sentences).to(RefCOCOg.DEVICE)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

        # bbox = torch.tensor(bbox).unsqueeze(0).to(device)

        print(f"Image shape: {image_features.shape}, Text shape: {text_features.shape}")

        # Combine image and text features and normalize
        product = torch.mul(image_features, text_features)
        out = torch.div(product, torch.norm(product, dim=1).reshape(-1, 1))

        # append bbox
        print(f"Output shape: {out.shape}")
        return out