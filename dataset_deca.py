import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy
from skimage.io import imread
from skimage.transform import estimate_transform, warp
import scipy.io
import face_alignment


class FAN(object):
    def __init__(self):
        self.model = face_alignment.FaceAlignment(
            face_alignment.LandmarksType._2D, flip_input=False
        )

    def run(self, image):
        """
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        """
        out = self.model.get_landmarks(image)
        if out is None:
            return [0], "kpt68"
        else:
            kpt = out[0].squeeze()
            left = np.min(kpt[:, 0])
            right = np.max(kpt[:, 0])
            top = np.min(kpt[:, 1])
            bottom = np.max(kpt[:, 1])
            bbox = [left, top, right, bottom]
            return bbox, "kpt68"


class TestData(Dataset):
    def __init__(
        self,
        imagepath_list,
        iscrop=True,
        crop_size=224,
        scale=1.25,
        just_return_image=True,
        face_detector=None,
    ):
        """
        testpath: folder, imagepath_list, image path, video path
        """
        self.imagepath_list = imagepath_list
        self.iscrop = iscrop
        self.crop_size = crop_size
        self.scale = scale
        self.just_return_image = just_return_image

        self.resolution_inp = crop_size
        self.face_detector = face_detector or FAN()

    def __len__(self):
        return len(self.imagepath_list)

    def bbox2point(self, left, right, top, bottom, type="bbox"):
        """bbox from detector and landmarks are different"""
        if type == "kpt68":
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array(
                [right - (right - left) / 2.0, bottom - (bottom - top) / 2.0]
            )
        elif type == "bbox":
            old_size = (right - left + bottom - top) / 2
            center = np.array(
                [
                    right - (right - left) / 2.0,
                    bottom - (bottom - top) / 2.0 + old_size * 0.12,
                ]
            )
        else:
            raise NotImplementedError
        return old_size, center

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(1, 1, 3)
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]

        h, w, _ = image.shape
        if self.iscrop:
            # provide kpt as txt file, or mat file (for AFLW2000)
            kpt_matpath = os.path.splitext(imagepath)[0] + ".mat"
            kpt_txtpath = os.path.splitext(imagepath)[0] + ".txt"
            if os.path.exists(kpt_matpath):
                kpt = scipy.io.loadmat(kpt_matpath)["pt3d_68"].T
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type="kpt68"
                )
            elif os.path.exists(kpt_txtpath):
                kpt = np.loadtxt(kpt_txtpath)
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type="kpt68"
                )
            else:
                bbox, bbox_type = self.face_detector.run(image)
                if len(bbox) < 4:
                    print("no face detected! run original image")
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                else:
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                old_size, center = self.bbox2point(
                    left, right, top, bottom, type=bbox_type
                )
            size = int(old_size * self.scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array(
            [[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]]
        )
        tform = estimate_transform("similarity", src_pts, DST_PTS)

        image = image / 255.0

        dst_image = warp(
            image,
            tform.inverse,
            output_shape=(self.resolution_inp, self.resolution_inp),
        )
        dst_image = dst_image.transpose(2, 0, 1)
        if self.just_return_image:
            return torch.tensor(dst_image).float()
        return {
            "image": torch.tensor(dst_image).float(),
            "imagename": imagename,
            "tform": torch.tensor(tform.params).float(),
            "original_image": torch.tensor(image.transpose(2, 0, 1)).float(),
        }
