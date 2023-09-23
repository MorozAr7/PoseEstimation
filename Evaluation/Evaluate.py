from CONFIG import *
from CoarsePoseEstimation.Evaluation import CoarsePoseEvaluation
from ObjectSegmentation.Evaluation import ObjectSegmentationEvaluation
from RefinedPoseEstimation.Evaluation import RefinedPoseEstimation
from DatasetRenderer.Renderer import DatasetRenderer
from Utils.IOUtils import IOUtils
import cv2
import os
import numpy as np


class Evaluation:
	def __init__(self):
		self.device = DEVICE
		self.coarse_pose_eval = CoarsePoseEvaluation(self.device)
		self.object_segmentation = ObjectSegmentationEvaluation(self.device)
		#self.pose_refinement = RefinedPoseEstimation(self.device)
		self.dataset_renderer = DatasetRenderer()
		self.video_path = "./Evaluation/ChassisTestVideos/Chassis_{}/".format(VIDEO)
		self.io = IOUtils()
		self.img_size = 224

	def load_video_frame(self, index):
		if len(str(index)) == 1:
			image_path = self.video_path + "images/" + "img_000{}.png".format(index)
			json_path = self.video_path + "inputs/" + "img_000{}.json".format(index)
			frame = cv2.imread(image_path)
			json_data = self.io.load_json_file(json_path)
		elif len(str(index)) == 2:
			image_path = self.video_path + "images/" + "img_00{}.png".format(index)
			json_path = self.video_path + "inputs/" + "img_00{}.json".format(index)
			frame = cv2.imread(image_path)
			json_data = self.io.load_json_file(json_path)
		elif len(str(index)) == 3:
			image_path = self.video_path + "images/" + "img_0{}.png".format(index)
			json_path = self.video_path + "inputs/" + "img_0{}.json".format(index)
			frame = cv2.imread(image_path)
			json_data = self.io.load_json_file(json_path)
		
		return frame, json_data
	
	def crop_image(self, frame, bbox):
		x_min, y_min, x_max, y_max = bbox
		width = x_max - x_min
		height = y_max - y_min
		center_x = int((x_min + x_max)/2)
		center_y = int((y_min + y_max)/2)
  
		max_size = max(width, height)
		new_x_min = center_x - max_size//2
		new_x_max = center_x + max_size//2
  
		new_y_min = center_y - max_size//2
		new_y_max = center_y + max_size//2

		return frame[new_y_min:new_y_max, new_x_min:new_x_max], [new_x_min, new_y_min, new_x_max, new_y_max]#frame[y_min:y_min + max_size, x_min:x_min + max_size]

	def evaluate_video(self):
		num_frames = len(os.listdir(self.video_path + "images"))
		#num_frames = range(115, 150)
		for index in range(num_frames):
			
			frame, json_data = self.load_video_frame(index)
			bbox = json_data[0]["bbox_modal"]
			cropepd_image, bbox = self.crop_image(frame, bbox)
			#cv2.imshow("img", cropepd_image)
			#cv2.waitKey()
			cropepd_image = cv2.resize(cropepd_image, (self.img_size, self.img_size))

			segmentation = self.object_segmentation.segment_image(np.expand_dims(cropepd_image, axis=0))
			
			pose_prediction = self.coarse_pose_eval.get_coarse_pose_estimate(np.expand_dims(cropepd_image, axis=0), np.expand_dims(segmentation, axis=0), torch.tensor(bbox).unsqueeze(0))[0]
			#refined_pose_prediction = self.pose_refinement.get_refined_pose(frame, np.expand_dims(pose_prediction, axis=0), bboxes=[bbox])
			#if refined_pose_prediction is None:
				#continue
			print("Coarse prediction", pose_prediction)
			print()
			print("Refined prediction", pose_prediction)
			img_dict_coarse = self.dataset_renderer.get_image(transformation_matrix=pose_prediction, image_black=True, image_background=False, constant_light=True)
			img_rendered_coarse = img_dict_coarse["ImageBlack"]
			mask = np.expand_dims(img_dict_coarse["Mask"], axis=-1)
			visualize_coarse = frame * (1 - mask) + img_rendered_coarse
			#pose_prediction[0:3, -1] = refined_pose_prediction[0, 0:3]
			#img_dict_refined = self.dataset_renderer.get_image(transformation_matrix=refined_pose_prediction, image_black=True, image_background=False, constant_light=True)
			#img_rendered_refined = img_dict_refined["ImageBlack"]
			#mask = np.expand_dims(img_dict_refined["Mask"], axis=-1)
			#visualize_refined = frame * (1 - mask) + img_rendered_refined
			cv2.imshow("video", np.vstack([visualize_coarse])/255)
			cv2.waitKey(1)


if __name__ == "__main__":
	eval = Evaluation()
	eval.evaluate_video()