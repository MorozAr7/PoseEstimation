from CnnModel import AutoencoderPoseEstimationModel
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from warnings import filterwarnings
import cv2
from CONFIG import *
filterwarnings("ignore")


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


def init_classification_model():
	weights_path = "./TrainedModels/CoarsePoseEstimationColorAug.pt"
	model = AutoencoderPoseEstimationModel()
	pretrained_dict = torch.load(weights_path, map_location="cpu")

	model_dict = model.state_dict()

	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	pretrained_dict_updated = {}
	for key, value in pretrained_dict.items():
		if "8" in key:
			pretrained_dict_updated[key] = torch.nn.init.xavier_uniform_(torch.rand((256, 32, 3, 3), requires_grad=True))
		else:
			pretrained_dict_updated[key] = value

	model.load_state_dict(pretrained_dict_updated)

	return model


def one_epoch(model, optimizer, dataloader, l1_loss_function, is_training=True, epoch=0):
	model.train() if is_training else model.eval()

	epoch_loss_l1_u_map = 0
	epoch_loss_l1_w_map = 0
	epoch_loss_l1_v_map = 0

	if is_training:
		for index, (image, mask, u_map, v_map, w_map) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			#print(image.shape, mask.shape, u_map.shape, v_map.shape, w_map.shape)
			optimizer.zero_grad()

			image = image.to(DEVICE)
			mask = mask.to(DEVICE)

			u_map = u_map.to(DEVICE)
			v_map = v_map.to(DEVICE)
			w_map = w_map.to(DEVICE)

			predictions = model(image)
			"""if epoch == 1:
				argmax_u = torch.argmax(predictions[0], dim=1)
				argmax_v = torch.argmax(predictions[1], dim=1)
				argmax_w = torch.argmax(predictions[2], dim=1)
				visualize_predictions = torch.cat([argmax_u.reshape(-1, 1, 224, 224) * mask, argmax_v.reshape(-1, 1, 224, 224) * mask, argmax_w.reshape(-1, 1, 224, 224) * mask, ], dim=1).reshape(
					-1,
					224
					* 3,
					224,
					1)
				print(argmax_w.shape, mask.shape)
				visualize_gt = torch.cat([u_map, v_map, w_map], dim=1).reshape(-1, 224 * 3, 224, 1)

				visualize = torch.cat([visualize_gt, visualize_predictions], dim=2)/255  # .resize(8, 128, 128, 1)
				visualize_np = visualize.detach().cpu().numpy()
				for i in range(BATCH_SIZE):
					cv2.imshow("img", image[i].permute(1, 2, 0).detach().cpu().numpy())
					cv2.waitKey(0)
					cv2.imshow("image", visualize_np[i])
					cv2.waitKey(0)"""

			loss_u = l1_loss_function(predictions[0] * mask, u_map)
			loss_v = l1_loss_function(predictions[1] * mask, v_map)
			loss_w = l1_loss_function(predictions[2] * mask, w_map)

			l1_batch_loss = loss_u + loss_w + loss_v

			total_loss = l1_batch_loss

			total_loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()

			epoch_loss_l1_u_map += loss_u.item()
			epoch_loss_l1_v_map += loss_v.item()
			epoch_loss_l1_w_map += loss_w.item()

		return epoch_loss_l1_u_map / (len(train_dataset)), epoch_loss_l1_v_map / (len(train_dataset)), epoch_loss_l1_w_map / (len(train_dataset))
	else:
		with torch.no_grad():
			for index, (image, mask, u_map, v_map, w_map) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)

				optimizer.zero_grad()

				image = image.to(DEVICE)
				mask = mask.to(DEVICE)
				u_map = u_map.to(DEVICE)
				v_map = v_map.to(DEVICE)
				w_map = w_map.to(DEVICE)

				predictions = model(image)

				loss_u = l1_loss_function(predictions[0] * mask, u_map)
				loss_v = l1_loss_function(predictions[1] * mask, v_map)
				loss_w = l1_loss_function(predictions[2] * mask, w_map)

				torch.cuda.empty_cache()

				epoch_loss_l1_u_map += loss_u.item()
				epoch_loss_l1_v_map += loss_v.item()
				epoch_loss_l1_w_map += loss_w.item()

			return epoch_loss_l1_u_map / (len(validation_dataset)), epoch_loss_l1_v_map / (len(validation_dataset)), epoch_loss_l1_w_map / (len(validation_dataset))


def main(model, optimizer, training_dataloader, validation_dataloader, loss_function):
	best_validation_accuracy = 0
	smallest_loss = float("inf")
	for epoch in range(1, 5000):

		since = time.time()
		change_learning_rate(optimizer, epoch)
		loss_u_t, loss_v_t, loss_w_t = one_epoch(model,
		                                         optimizer,
		                                         training_dataloader,
		                                         loss_function,
		                                         is_training=True,
		                                         epoch=epoch
		                                         )
		loss_u_v, loss_v_v, loss_w_v = one_epoch(model,
		                                         optimizer,
		                                         validation_dataloader,
		                                         loss_function,
		                                         is_training=False
		                                         )

		print("Epoch: {}, "
		      "Train loss u: {}, "
		      "Train loss v: {}, "
		      "Train loss w: {}, "
		      "Valid loss u: {}, "
		      "Valid loss v: {}, "
		      "Valid loss w: {}".format(epoch,
		                                loss_u_t,
		                                loss_v_t,
		                                loss_w_t,
		                                loss_u_v,
		                                loss_v_v,
		                                loss_w_v))

		print("EPOCH RUNTIME", time.time() - since)

		if loss_u_v + loss_v_v + loss_w_v < smallest_loss:
			smallest_loss = loss_u_v + loss_v_v + loss_w_v
			print("SAVING MODEL")
			torch.save(model.state_dict(), "{}.pt".format("./TrainedModels/CoarsePoseEstimatorModel"))
			print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model = AutoencoderPoseEstimationModel()#init_classification_model()
	model.load_state_dict(torch.load("./TrainedModels/CoarsePoseEstimatorModel.pt", map_location="cpu"))
	model.to(DEVICE)

	optimizer = torch.optim.Adam(lr=LR, params=model.parameters())
	loss_function = nn.CrossEntropyLoss(reduction="sum")

	train_dataset = Dataset(subset="Training", num_images=100000, data_augmentation=PoseEstimationAugmentation)
	validation_dataset = Dataset(subset="Validation", num_images=10000, data_augmentation=None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

	main(model, optimizer, training_dataloader, validation_dataloader, loss_function)
