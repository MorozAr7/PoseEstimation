from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import torch
import sys
import numpy as np
from CnnModel import PoseRefinementNetwork
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from Utils.IOUtils import IOUtils
from Utils.ConvUtils import init_weights, change_learning_rate
from RefinedPoseEstimation.LossFunction import ProjectionLoss
from torch.utils.data import DataLoader
import torch.nn as nn
import warnings
import time
import cv2
from DatasetRenderer.Renderer import DatasetRenderer
warnings.filterwarnings("ignore")


def one_epoch(pose_refiner_model, optimizer, dataloader, loss_function, is_training=True, epoch=0):
	pose_refiner_model.train() if is_training else pose_refiner_model.eval()
	epoch_loss_rotation = 0
	epoch_loss_translation_z = 0
	epoch_loss_translation_xy = 0

	if is_training:
		for index, (images_real, images_rendered, T_target, T_coarse) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			optimizer.zero_grad()
			images_real = images_real.to(DEVICE)
			images_rendered = images_rendered.to(DEVICE)
	
			visualize = 255 * torch.cat([images_real, images_rendered], dim=-1).permute(0, 2, 3, 1).detach().cpu().numpy()
			cv2.imwrite("image_test_10.png".format(epoch), visualize[0].astype(np.uint8))
   
			T_target = T_target.to(DEVICE)
			T_coarse = T_coarse.to(DEVICE)

			predicted_translation, predicted_rotation = pose_refiner_model(images_real, images_rendered)

			loss_data = loss_function(predicted_translation, predicted_rotation, T_coarse, T_target)
			loss_R, loss_xy, loss_z, loss_total = loss_data["LossR"], loss_data["LossXY"], loss_data["LossZ"], loss_data["LossTotal"]
   
			if LOSS_TYPE == 0:
				loss = LOSS_WEIGHTS["R"] * loss_R + LOSS_WEIGHTS["XY"] * loss_xy + LOSS_WEIGHTS["Z"] * loss_z
				epoch_loss_rotation += loss_total.item()
				epoch_loss_translation_xy += loss_xy.item()
				epoch_loss_translation_z += loss_z.item()
			elif LOSS_TYPE == 1:
				loss = loss_total
				epoch_loss_rotation += loss_total.item()
			elif LOSS_TYPE == 2:
				loss = loss_R + loss_xy + loss_z + loss_total
				epoch_loss_rotation += loss_total.item()
				epoch_loss_translation_xy += loss_xy.item()
				epoch_loss_translation_z += loss_z.item()
			#cv2.imshow("image", visualize[0]/255)
			#cv2.waitKey(0)
			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			
		return epoch_loss_rotation / len(train_dataset), epoch_loss_translation_xy / len(train_dataset), epoch_loss_translation_z/ len(train_dataset)
	else:

		with torch.no_grad():
			for index, (images_real, images_rendered, T_target, T_coarse) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)
				optimizer.zero_grad()
				images_real = images_real.to(DEVICE)
				images_rendered = images_rendered.to(DEVICE)
	
				T_target = T_target.to(DEVICE)
				T_coarse = T_coarse.to(DEVICE)

				predicted_translation, predicted_rotation = pose_refiner_model(images_real, images_rendered)

				loss_data = loss_function(predicted_translation, predicted_rotation, T_coarse, T_target)
				loss_R, loss_xy, loss_z, loss_total = loss_data["LossR"], loss_data["LossXY"], loss_data["LossZ"], loss_data["LossTotal"]
				
				if LOSS_TYPE == 0:
					loss = LOSS_WEIGHTS["R"] * loss_R + LOSS_WEIGHTS["XY"] * loss_xy + LOSS_WEIGHTS["Z"] * loss_z
					epoch_loss_rotation += loss_total.item()
					epoch_loss_translation_xy += loss_xy.item()
					epoch_loss_translation_z += loss_z.item()
				elif LOSS_TYPE == 1:
					loss = loss_total
					epoch_loss_rotation += loss_total.item()
				elif LOSS_TYPE == 2:
					loss = loss_R + loss_xy + loss_z + loss_total
					epoch_loss_rotation += loss_total.item()
					epoch_loss_translation_xy += loss_xy.item()
					epoch_loss_translation_z += loss_z.item()
     
				torch.cuda.empty_cache()

			return epoch_loss_rotation / len(validation_dataset), epoch_loss_translation_xy / len(validation_dataset), epoch_loss_translation_z / len(validation_dataset)


def main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, loss_function) -> None:

	smallest_loss = float("inf")
	for epoch in range(1, NUM_EPOCHS):
		since: float = time.time()
		change_learning_rate(optimizer, epoch, LR_DECAY_EPOCHS, LR_DECAY_FACTOR)
		train_l_rotation, train_l_xy, train_l_z = one_epoch(
		                           pose_refiner_model,
		                           optimizer,
		                           training_dataloader,
		                           loss_function,
		                           is_training=True,
		                           epoch=epoch
		                           )
		valid_l_rotation, valid_l_xy, valid_l_z = one_epoch(
		                           pose_refiner_model,
		                           optimizer,
		                           validation_dataloader,
		                           loss_function,
		                           is_training=False
		                           )
		print("Epoch: {}, Train rotation: {}, Train xy: {}, Train z: {}, Valid rotation: {}, Valid xy: {}, Valid z: {}"
		      .format(epoch, train_l_rotation, train_l_xy, train_l_z, valid_l_rotation, valid_l_xy, valid_l_z))
		print("EPOCH RUNTIME", time.time() - since)

		if valid_l_rotation + valid_l_xy + valid_l_z < smallest_loss:
			smallest_loss = valid_l_rotation + valid_l_xy + valid_l_z
		print("SAVING MODEL")
		torch.save(pose_refiner_model.state_dict(), "{}.pt".format("./TrainedModels/RefinedPoseEstimationModelProjection2DfasterLRreduction"))
		print("MODEL WAS SUCCESSFULLY SAVED!")



if __name__ == "__main__":
	torch.autograd.set_detect_anomaly(True)
 
	dataset_renderer = DatasetRenderer()
	pose_refiner_model = PoseRefinementNetwork().to(DEVICE).apply(init_weights)
	pose_refiner_model.load_state_dict(torch.load("./TrainedModels/RefinedPoseEstimationModelProjection2DfasterLRreduction.pt", map_location="cpu"))
	
	io = IOUtils()
	point_cloud = io.load_numpy_file(MAIN_DIR_PATH + "/DatasetRenderer/Models3D/Chassis/SparcePointCloud5k.npy")
	point_cloud_torch = torch.tensor(point_cloud).float().to(DEVICE)
 
	optimizer = torch.optim.Adam(lr=LR, params=pose_refiner_model.parameters())
	l1_loss_function = ProjectionLoss(point_cloud=point_cloud_torch, device=DEVICE, projection_type=PROJECTION_TYPE_LOSS, disentangle=LOSS_TYPE)
 
	subset = "Training"
	train_dataset = Dataset(subset, NUM_DATA[subset], dataset_renderer, PoseEstimationAugmentation if USE_AUGMENTATION else None)
	subset = "Validation"
	validation_dataset = Dataset(subset, NUM_DATA[subset], dataset_renderer, None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=32)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=32)

	main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, l1_loss_function)
	
