from CONFIG import *
from CnnModel import AutoencoderPoseEstimationModel
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from warnings import filterwarnings
import cv2
filterwarnings("ignore")

def init_weights(m):
	if type(m) in [nn.Conv2d, nn.ConvTranspose2d]:
		torch.nn.init.xavier_uniform_(m.weight)
	elif type(m) in [nn.BatchNorm2d]:
		torch.nn.init.normal_(m.weight.data, 1.0, 2.0)
		torch.nn.init.constant_(m.bias.data, 0)


def change_learning_rate(optimizer, epoch):
	epochs_to_change = list(range(25, 500, 25))
	if epoch in epochs_to_change:
		optimizer.param_groups[0]["lr"] /= 1.5


def init_classification_model():
	weights_path = "./CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModel.pt"
	model = AutoencoderPoseEstimationModel()
	pretrained_dict = torch.load(weights_path, map_location="cpu")

	model_dict = model.state_dict()

	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	pretrained_dict_updated = {}
	for key, value in pretrained_dict.items():
		if "ResLayer" in key:
			#print(key)
			pass
			#pretrained_dict_updated[key] = torch.nn.init.xavier_uniform_(torch.rand((256, 32, 3, 3), requires_grad=True))
		else:
			print("param_loaded")
			pretrained_dict_updated[key] = value

	model.load_state_dict(pretrained_dict_updated, strict=False)

	return model


def one_epoch(model, optimizer, dataloader, loss_function, is_training=True, epoch=0):
	model.train() if is_training else model.eval()

	epoch_loss_l1_u_map = 0
	epoch_loss_l1_w_map = 0
	epoch_loss_l1_v_map = 0

	if is_training:
		for index, (image, mask, u_map, v_map, w_map) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
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

			loss_u = loss_function(predictions[0] * mask, u_map)
			loss_v = loss_function(predictions[1] * mask, v_map)
			loss_w = loss_function(predictions[2] * mask, w_map)

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

				loss_u = loss_function(predictions[0] * mask, u_map)
				loss_v = loss_function(predictions[1] * mask, v_map)
				loss_w = loss_function(predictions[2] * mask, w_map)

				torch.cuda.empty_cache()

				epoch_loss_l1_u_map += loss_u.item()
				epoch_loss_l1_v_map += loss_v.item()
				epoch_loss_l1_w_map += loss_w.item()

			return epoch_loss_l1_u_map / (len(validation_dataset)), epoch_loss_l1_v_map / (len(validation_dataset)), epoch_loss_l1_w_map / (len(validation_dataset))


def main(model, optimizer, training_dataloader, validation_dataloader, loss_function):
	smallest_loss = float("inf")
	for epoch in range(1, NUM_EPOCHS):

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

		print(f"Epoch: {epoch}, "
		      f"Train loss u: {loss_u_t}, "
		    	f"Train loss v: {loss_v_t}, "
		      f"Train loss w: {loss_w_t}, "
		      f"Valid loss u: {loss_u_v}, "
		      f"Valid loss v: {loss_v_v}, "
		      f"Valid loss w: {loss_w_v}")

		print("EPOCH RUNTIME", time.time() - since)

		if loss_u_v + loss_v_v + loss_w_v < smallest_loss and SAVE_MODEL:
			smallest_loss = loss_u_v + loss_v_v + loss_w_v
		print("SAVING MODEL")
		torch.save(model.state_dict(), "{}.pt".format("./TrainedModels/CoarsePoseEstimatorModelNewMeshOrientationNewResLayer"))
		print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model = init_classification_model()#AutoencoderPoseEstimationModel()
	#model.load_state_dict(torch.load("./TrainedModels/CoarsePoseEstimatorModelNewMeshOrientation.pt.pt", map_location="cpu"))
	model.to(DEVICE)

	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
	loss_function = nn.CrossEntropyLoss(reduction="sum")

	train_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[0], 
			 				num_images=list(SUBSET_NUM_DATA.values())[0], 
			 				data_augmentation=PoseEstimationAugmentation)
	
	validation_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[1], 
			      				num_images=list(SUBSET_NUM_DATA.values())[1], 
				  				data_augmentation=None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=32)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=32)

	main(model, optimizer, training_dataloader, validation_dataloader, loss_function)
