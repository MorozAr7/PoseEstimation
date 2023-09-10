import gc
from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import torch
import sys
import os
import subprocess
import signal
import numpy as np
from CnnModel import PoseRefinementNetwork
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from Utils.IOUtils import IOUtils
from Utils.Losses import ProjectionLoss
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import cv2
from DatasetRenderer.Renderer import DatasetRenderer

def init_weights(m) -> None:
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optimizer, epoch) -> None:
	epochs_to_change = list(range(50, 5000, 50))
	if epoch in epochs_to_change:
		optimizer.param_groups[0]["lr"] /= 2


def one_epoch(pose_refiner_model, optimizer, dataloader, l1_loss_function, is_training=True, epoch=0):
	pose_refiner_model.train() if is_training else pose_refiner_model.eval()
	epoch_loss_rotation = 0
	epoch_loss_translation_z = 0
	epoch_loss_translation_xy = 0

	if is_training:
		for index, (images, refinement_images1, T_target, T_coarse) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			optimizer.zero_grad()
			images = images.to(DEVICE)
			refinement_images1 = refinement_images1.to(DEVICE)
   
			visualize = 255 * torch.cat([images, refinement_images1], dim=-1).permute(0, 2, 3, 1).detach().cpu().numpy()
			cv2.imwrite("image_test_1.png".format(epoch), visualize[0].astype(np.uint8))
   
			T_target = T_target.to(DEVICE)
			T_coarse = T_coarse.to(DEVICE)

			predicted_translation, predicted_rotation = pose_refiner_model(torch.cat([images, refinement_images1], dim=1))
			t_coarse = T_coarse[..., 0:3, -1]
			t_target = T_target[..., 0:3, -1]

			loss_xy, loss_z, loss_R = l1_loss_function(predicted_translation, predicted_rotation, T_coarse, T_target)
			disentangled_loss = 1 * loss_z + 1 * loss_xy + 1 * loss_R
   
			disentangled_loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()

			epoch_loss_rotation += loss_R.item()
			epoch_loss_translation_xy += torch.sum(torch.abs((predicted_translation[..., :2] + t_coarse[..., 0:2] / t_coarse[..., 2:3]) * t_target[..., 2:3] - t_target[..., :2])).item()
			epoch_loss_translation_z += torch.sum(torch.abs((t_coarse[..., -1] * predicted_translation[..., 2] - t_target[..., -1]))).item()

		return epoch_loss_rotation / len(train_dataset), epoch_loss_translation_xy / len(train_dataset), epoch_loss_translation_z/ len(train_dataset)
	else:

		with torch.no_grad():
			for index, (images, refinement_images1, T_target, T_coarse) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)
				optimizer.zero_grad()
				images = images.to(DEVICE)
				refinement_images1 = refinement_images1.to(DEVICE)
	
				#visualize = 255 * torch.cat([images, refinement_images1], dim=-1).permute(0, 2, 3, 1).detach().cpu().numpy()
				#cv2.imwrite("image_test_1.png".format(epoch), visualize[0].astype(np.uint8))
	
				T_target = T_target.to(DEVICE)
				T_coarse = T_coarse.to(DEVICE)

				predicted_translation, predicted_rotation = pose_refiner_model(torch.cat([images, refinement_images1], dim=1))
				t_coarse = T_coarse[..., 0:3, -1]
				t_target = T_target[..., 0:3, -1]

				loss_xy, loss_z, loss_R = l1_loss_function(predicted_translation, predicted_rotation, T_coarse, T_target)
				disentangled_loss = loss_z + loss_xy + loss_R

				torch.cuda.empty_cache()

				epoch_loss_rotation += loss_R.item()
				epoch_loss_translation_xy += torch.sum(torch.abs((predicted_translation[..., :2] + t_coarse[..., 0:2] / t_coarse[..., 2:3]) * t_target[..., 2:3] - t_target[..., :2])).item()
				epoch_loss_translation_z += torch.sum(torch.abs((t_coarse[..., -1] * predicted_translation[..., 2] - t_target[..., -1]))).item()


			return epoch_loss_rotation / len(validation_dataset), epoch_loss_translation_xy / len(validation_dataset), epoch_loss_translation_z / len(validation_dataset)


def main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, l1_loss_function) -> None:

	smallest_loss = float("inf")
	for epoch in range(1, 1000):
		# print("START")
		since: float = time.time()
		change_learning_rate(optimizer, epoch)
		train_l_rotation, train_l_xy, train_l_z = one_epoch(
		                           pose_refiner_model,
		                           optimizer,
		                           training_dataloader,
		                           l1_loss_function,
		                           is_training=True,
		                           epoch=epoch
		                           )
		valid_l_rotation, valid_l_xy, valid_l_z = one_epoch(
		                           pose_refiner_model,
		                           optimizer,
		                           validation_dataloader,
		                           l1_loss_function,
		                           is_training=False
		                           )
		print("Epoch: {}, Train rotation: {}, Train xy: {}, Train z: {}, Valid rotation: {}, Valid xy: {}, Valid z: {}"
		      .format(epoch, train_l_rotation, train_l_xy, train_l_z, valid_l_rotation, valid_l_xy, valid_l_z))
		print("EPOCH RUNTIME", time.time() - since)

		if valid_l_rotation + valid_l_xy + valid_l_z < smallest_loss:
			smallest_loss = valid_l_rotation + valid_l_xy + valid_l_z
		print("SAVING MODEL")
		torch.save(pose_refiner_model.state_dict(), "{}.pt".format("./TrainedModels/RefinedPoseEstimationModel"))
		print("MODEL WAS SUCCESSFULLY SAVED!")
		"""pid = os.getpid()
		print("THE CURRENT PROCESS WITH PID : {} HAS BEEN KILLED".format(pid))
		subprocess.run(["python3", "TrainModel.py"])
		os.kill(pid, signal.SIGKILL)"""


if __name__ == "__main__":
	torch.autograd.set_detect_anomaly(True)
	dataset_renderer = DatasetRenderer()
	pose_refiner_model = PoseRefinementNetwork().to(DEVICE).apply(init_weights)
	pose_refiner_model.load_state_dict(torch.load("./TrainedModels/RefinedPoseEstimationModel.pt", map_location="cpu"))
	
	io = IOUtils()
	point_cloud = io.load_numpy_file(MAIN_DIR_PATH + "/DatasetRenderer/Models3D/Chassis/SparcePointCloud5k.npy")
	point_cloud_torch = torch.tensor(point_cloud).float().to(DEVICE)
 
	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=pose_refiner_model.parameters())
	l1_loss_function = ProjectionLoss(point_cloud=point_cloud_torch, device=DEVICE)#nn.L1Loss(reduction="sum")
	train_dataset = Dataset("Training", 10000, dataset_renderer, PoseEstimationAugmentation)
	validation_dataset = Dataset("Validation", 2000, dataset_renderer, None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

	main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, l1_loss_function)
	
