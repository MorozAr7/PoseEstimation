from CONFIG import *
from ObjectSegmentation.CnnModel import UnetSegmentation
import torch
import cv2
from Utils.DataAugmentationUtils import NormalizeToTensor


class ObjectSegmentationEvaluation:
	def __init__(self, device):
		self.segmentation_model = UnetSegmentation()
		self.threshold = 0.5
		self.input_size = 224
		self.device = device
		self.load_model_weights()

	def load_model_weights(self):
		self.segmentation_model.load_state_dict(torch.load(MAIN_DIR_PATH + "/ObjectSegmentation/TrainedModels/ObjectSegmentationModel.pt",
		                                                      map_location="cpu"))
		self.segmentation_model.eval()
		self.segmentation_model.to(self.device)

	@staticmethod
	def normalize_convert_to_tensor(image):
		return NormalizeToTensor(image=image)["image"]

	def convert_images_to_tensors(self, images):
		batch_tensor = torch.tensor([])
		for index in range(images.shape[0]):
			image = images[index, ...]
			image_tensor = self.normalize_convert_to_tensor(image)
			batch_tensor = torch.cat([batch_tensor, image_tensor], dim=0)
		return batch_tensor.unsqueeze(0)

	def segment_image(self, images):
		batch_tensor = self.convert_images_to_tensors(images).to(self.device)
		with torch.no_grad():
			mask = (self.segmentation_model(batch_tensor) > self.threshold).float()
		print(mask.shape)
		return mask.permute(0, 2, 3, 1).cpu().numpy()[0]
	

if __name__ == "__main__":
	obj_segm = ObjectSegmentationEvaluation("mps")
