import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

PoseEstimationAugmentation = A.Compose([
		A.ChannelShuffle(p=0.75),

		A.ColorJitter(brightness=(0.75, 1.75), contrast=(0.5, 2), saturation=(1, 5), hue=(0, 0), always_apply=False, p=0.8),
		A.OneOf([
			A.GaussNoise(var_limit=(25.0, 40.0), mean=0, per_channel=True, always_apply=False, p=1),
			A.MultiplicativeNoise(multiplier=(0.925, 1.075), per_channel=True, elementwise=True, always_apply=False, p=1),
			A.ISONoise(color_shift=(0.03, 0.06), intensity=(0.1, 0.4), always_apply=False, p=1)

		], p=0.4),
])


ObjectSegmentationAugmentation = A.Compose([
		A.ChannelShuffle(p=0.75),
		A.ColorJitter(brightness=(0.75, 1.75), contrast=(0.5, 2), saturation=(1, 5), hue=(0, 0), always_apply=False, p=0.8),
		A.HorizontalFlip(p=0.5),
		A.VerticalFlip(p=0.5),
		A.Rotate(p=1, limit=(180, 180), border_mode=cv2.BORDER_REPLICATE),
		A.OneOf([
			A.GaussNoise(var_limit=(25.0, 40.0), mean=0, per_channel=True, always_apply=True, p=1),
			A.MultiplicativeNoise(multiplier=(0.925, 1.075), per_channel=True, elementwise=True, always_apply=True, p=1),
			A.ISONoise(color_shift=(0.03, 0.06), intensity=(0.1, 0.4), always_apply=False, p=1)

		], p=0.25),
])


NormalizeToTensor = A.Compose([
		A.Normalize((0, 0, 0), (1, 1, 1), max_pixel_value=255),
		ToTensorV2()
])


