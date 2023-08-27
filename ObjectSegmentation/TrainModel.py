import torch
from CONFIG import *
sys.path.insert(0, MAIN_DIR_PATH)
import numpy as np
from CnnModel import UnetSegmentation
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import ObjectSegmentationAugmentation
from torch.utils.data import DataLoader
import torch.nn as nn
import time
import cv2

from warnings import filterwarnings

filterwarnings("ignore")


def get_mask_iou(predictions, target):
	predictions = (predictions > 0.5).float()
	intersection = target * predictions
	union = target + predictions - intersection
	sum_union = torch.sum(union, dim=(1, 2, 3))
	sum_intersection = torch.sum(intersection, dim=(1, 2, 3))

	iou = sum_intersection / sum_union

	return iou.unsqueeze(1)


def init_weights(m):
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optimizer, epoch):
	epochs_to_change = list(range(10, 500, 10))
	if epoch in epochs_to_change:
		optimizer.param_groups[0]["lr"] /= 2


def one_epoch(model, optimizer, dataloader, bce_loss_function, is_training=True, epoch=0):
	model.train() if is_training else model.eval()

	epoch_loss = 0
	epoch_iou = torch.tensor([]).to(DEVICE)
	if is_training:
		for index, (images, labels) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)

			optimizer.zero_grad()
			images = images.to(DEVICE)
			labels = labels.to(DEVICE)

			predictions = model(images)
			iou = get_mask_iou(predictions, labels)
			epoch_iou = torch.cat([epoch_iou, iou], dim=0)
			loss = bce_loss_function(predictions, labels)
			"""visualize = torch.cat([labels.repeat(1, 3, 1, 1), (predictions.repeat(1, 3, 1, 1) > 0.5).float(), images], dim=3).permute(0, 2, 3, 1).detach().cpu().numpy()
			for i in range(BATCH_SIZE):
				print(iou[i])
				cv2.imshow("visualize", visualize[i, ...])
				cv2.waitKey(0)"""
			loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			epoch_loss += loss.item()

		return epoch_loss / (len(train_dataset)), torch.sum(epoch_iou) / (len(train_dataset))
	else:
		with torch.no_grad():
			for index, (images, labels) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)

				optimizer.zero_grad()
				images = images.to(DEVICE)
				labels = labels.to(DEVICE)

				predictions = model(images)
				iou = get_mask_iou(predictions, labels)
				epoch_iou = torch.cat([epoch_iou, iou], dim=0)
				loss = bce_loss_function(predictions, labels)

				torch.cuda.empty_cache()

				epoch_loss += loss.item()

			return epoch_loss / (len(validation_dataset)), torch.sum(epoch_iou) / (len(validation_dataset))


def main(model, optimizer, training_dataloader, validation_dataloader, loss_function):
	best_validation_accuracy = 0
	smallest_loss = float("inf")
	for epoch in range(1, 5000):

		since = time.time()
		change_learning_rate(optimizer, epoch)
		train_loss_bce, train_iou = one_epoch(model,
		                                      optimizer,
		                                      training_dataloader,
		                                      loss_function,
		                                      is_training=True,
		                                      epoch=epoch
		                                      )
		valid_loss_bce, valid_iou = one_epoch(model,
		                                      optimizer,
		                                      validation_dataloader,
		                                      loss_function,
		                                      is_training=False
		                                      )

		print("Epoch: {}, Train loss: {}, Valid loss: {}, Train iou: {}, Valid iou: {}".format(epoch, train_loss_bce, valid_loss_bce, train_iou, valid_iou))
		print("EPOCH RUNTIME", time.time() - since)

		if valid_iou > best_validation_accuracy:
			best_validation_accuracy = valid_iou
			print("SAVING MODEL")
			torch.save(model.state_dict(), "{}.pt".format("./TrainedModels/ObjectSegmentationModel"))
			print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model = UnetSegmentation().to(DEVICE).apply(init_weights)
	model.load_state_dict(torch.load("./TrainedModels/ObjectSegmentationModel.pt", map_location="cpu"))

	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
	loss_function = nn.BCELoss(reduction="sum")

	train_dataset = Dataset(subset="Training", num_images=100000, data_augmentation=ObjectSegmentationAugmentation)
	validation_dataset = Dataset(subset="Validation", num_images=10000, data_augmentation=None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

	main(model, optimizer, training_dataloader, validation_dataloader, loss_function)
