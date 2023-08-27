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
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import cv2
from DatasetRenderer.Renderer import DatasetRenderer


def init_weights(m):
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d, nn.Linear]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optimizer, epoch):
	epochs_to_change = list(range(20, 500, 20))
	if epoch in epochs_to_change:
		optimizer.param_groups[0]["lr"] /= 2


def one_epoch(pose_refiner_model, optimizer, dataloader, l1_loss_function, is_training=True, epoch=0):
	pose_refiner_model.train() if is_training else pose_refiner_model.eval()
	epoch_loss_rotation = 0
	epoch_loss_translation_z = 0
	epoch_loss_translation_xy = 0

	if is_training:
		for index, (images, refinement_images, angle_target, t_target, angle_coarse, t_coarse) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			optimizer.zero_grad()
			images = images.to(DEVICE)
			refinement_images = refinement_images.to(DEVICE)

			visualize = 255 * torch.cat([images, refinement_images], dim=-1).permute(0, 2, 3, 1).detach().cpu().numpy()
			cv2.imwrite("image_test_{}.png".format(epoch), visualize[0].astype(np.uint8))
			angle_target = angle_target.to(DEVICE)
			t_target = t_target.to(DEVICE)
			angle_coarse = angle_coarse.to(DEVICE)
			t_coarse = t_coarse.to(DEVICE)

			refined = pose_refiner_model(images, refinement_images, torch.cat([t_coarse, angle_coarse], dim=1))

			"""for i in range(16):
				print(torch.cat([t_coarse, angle_coarse], dim=1)[i])
				print(refined[i])
				print(torch.cat([t_target, angle_target], dim=1)[i])
				print("-----------------------------------------------------")
				cv2.imshow("visualize", visualize[i, ...])
				cv2.waitKey(0)"""
			loss_rotation = l1_loss_function(refined[..., 3:], angle_target)
			loss_z = l1_loss_function(refined[..., 2:3], t_target[..., 2:3])
			loss_xy = l1_loss_function(refined[..., :2], t_target[..., :2])
			loss = loss_rotation + loss_z + loss_xy

			loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			epoch_loss_rotation += loss_rotation.item()
			epoch_loss_translation_xy += loss_xy.item()
			epoch_loss_translation_z += loss_z.item()

		return epoch_loss_rotation / len(train_dataset), epoch_loss_translation_xy / len(train_dataset), epoch_loss_translation_z / len(train_dataset)
	else:

		with torch.no_grad():
			for index, (images, refinement_images, angle_target, t_target, angle_coarse, t_coarse) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)
				optimizer.zero_grad()
				images = images.to(DEVICE)
				refinement_images = refinement_images.to(DEVICE)

				angle_target = angle_target.to(DEVICE)
				t_target = t_target.to(DEVICE)
				angle_coarse = angle_coarse.to(DEVICE)
				t_coarse = t_coarse.to(DEVICE)

				refined = pose_refiner_model(images, refinement_images, torch.cat([t_coarse, angle_coarse], dim=1))

				loss_rotation = l1_loss_function(refined[..., 3:], angle_target)
				loss_z = l1_loss_function(refined[..., 2:3], t_target[..., 2:3])
				loss_xy = l1_loss_function(refined[..., :2], t_target[..., :2])

				torch.cuda.empty_cache()

				epoch_loss_rotation += loss_rotation.item()
				epoch_loss_translation_xy += loss_xy.item()
				epoch_loss_translation_z += loss_z.item()

			return epoch_loss_rotation / len(validation_dataset), epoch_loss_translation_xy / len(validation_dataset), epoch_loss_translation_z / len(validation_dataset)


def main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, l1_loss_function):
	best_validation_accuracy = 0
	smallest_loss_translation = float("inf")
	smallest_loss_rotation = float("inf")
	for epoch in range(1, 5000):
		# print("START")
		since = time.time()
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

		if valid_l_rotation + valid_l_xy + valid_l_z < smallest_loss_rotation:
			smallest_loss_rotation = valid_l_rotation + valid_l_xy + valid_l_z
			print("SAVING MODEL")
			torch.save(pose_refiner_model.state_dict(), "{}.pt".format("./TrainedModels/RefinedPoseEstimatorModel"))
			print("MODEL WAS SUCCESSFULLY SAVED!")
		pid = os.getpid()
		print("THE CURRENT PROCESS WITH PID : {} HAS BEEN KILLED".format(pid))
		subprocess.run(["python3", "TrainModel.py"])
		os.kill(pid, signal.SIGKILL)


if __name__ == "__main__":
	dataset_renderer = DatasetRenderer()
	pose_refiner_model = PoseRefinementNetwork().to(DEVICE)#.apply(init_weights)
	pose_refiner_model.load_state_dict(torch.load("./TrainedModels/RefinedPoseEstimatorModel.pt", map_location="cpu"))

	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=pose_refiner_model.parameters())
	l1_loss_function = nn.L1Loss(reduction="sum")
	train_dataset = Dataset("Training", 100000, dataset_renderer, PoseEstimationAugmentation)
	validation_dataset = Dataset("Validation", 10000, dataset_renderer, None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

	main(pose_refiner_model, optimizer, training_dataloader, validation_dataloader, l1_loss_function)
