from CONFIG import *
from CnnModel import AutoencoderPoseEstimationModel
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from torch.utils.data import DataLoader
from LossFunction import MultiscaleSsimLossFunction
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
		optimizer.param_groups[0]["lr"] /= 1.75


def init_classification_model():
	weights_path = MAIN_DIR_PATH + "CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModelNewMeshOrientationNewResLayer.pt"
	model = AutoencoderPoseEstimationModel()
	pretrained_dict = torch.load(weights_path, map_location="cpu")

	model_dict = model.state_dict()

	pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

	pretrained_dict_updated = {}
	for key, value in pretrained_dict.items():
		if "Conv8.Conv.weight" in key:
			pretrained_dict_updated[key] = torch.nn.init.xavier_uniform_(torch.rand((1, 32, 3, 3), requires_grad=True))
		else:
			pretrained_dict_updated[key] = value

	model.load_state_dict(pretrained_dict_updated, strict=False)

	return model


def one_epoch(model, optimizer, dataloader, loss_functions, is_training=True, epoch=0):
	model.train() if is_training else model.eval()
	epoch_loss_l1_u_map = 0
	epoch_loss_l1_w_map = 0
	epoch_loss_l1_v_map = 0
	l1_loss_function = loss_functions[0]
	ssim_loss_function = loss_functions[1]
	if is_training:
		for index, (image, mask, u_map, v_map, w_map) in enumerate(dataloader):
			print("BATCH TRAINING: ", index)
			optimizer.zero_grad()

			image = image.to(DEVICE)
			mask = mask.to(DEVICE).reshape(-1, 1, 224, 224)

			u_map = u_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255
			v_map = v_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255
			w_map = w_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255

			predictions = model(image)

			l1_u = l1_loss_function(predictions[0] * mask, u_map)
			l1_v = l1_loss_function(predictions[1] * mask, v_map)
			l1_w = l1_loss_function(predictions[2] * mask, w_map)
			
			ssim_loss_u = ssim_loss_function.get_multiscale_structural_sim_loss(predictions[0] * mask, u_map)
			ssim_loss_v = ssim_loss_function.get_multiscale_structural_sim_loss(predictions[1] * mask, v_map)
			ssim_loss_w = ssim_loss_function.get_multiscale_structural_sim_loss(predictions[2] * mask, w_map)
   
			grad_ssim_loss_u = ssim_loss_function.get_gradient_ssim_loss(predictions[0] * mask, u_map)
			grad_ssim_loss_v = ssim_loss_function.get_gradient_ssim_loss(predictions[1] * mask, v_map)
			grad_ssim_loss_w = ssim_loss_function.get_gradient_ssim_loss(predictions[2] * mask, w_map)

			l1_total = l1_u + l1_v + l1_w
			ssim_total = ssim_loss_u + ssim_loss_v + ssim_loss_w
			grad_total = grad_ssim_loss_u + grad_ssim_loss_v + grad_ssim_loss_w
			total_loss = 4 * l1_total + ssim_total + grad_total

			total_loss.backward()
			optimizer.step()
			torch.cuda.empty_cache()
			mask = mask.reshape(-1, 1, 224, 224)

			diff_u = torch.sum(torch.abs(predictions[0] * mask * 255 - 255 * u_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
			diff_v = torch.sum(torch.abs(predictions[1] * mask * 255 - 255 * v_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
			diff_w = torch.sum(torch.abs(predictions[2] * mask * 255 - 255 * w_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
   
			epoch_loss_l1_u_map += torch.sum(diff_u).item()
			epoch_loss_l1_v_map += torch.sum(diff_v).item()
			epoch_loss_l1_w_map += torch.sum(diff_w).item()

		return epoch_loss_l1_u_map / (len(train_dataset)), epoch_loss_l1_v_map / (len(train_dataset)), epoch_loss_l1_w_map / (len(train_dataset))
	else:
		with torch.no_grad():
			for index, (image, mask, u_map, v_map, w_map) in enumerate(dataloader):
				print("BATCH VALIDATION: ", index)

				optimizer.zero_grad()

				image = image.to(DEVICE)
				
				mask = mask.to(DEVICE).reshape(-1, 1, 224, 224)

				u_map = u_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255
				v_map = v_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255
				w_map = w_map.to(DEVICE).reshape(-1, 1, 224, 224) / 255

				predictions = model(image)

				torch.cuda.empty_cache()
				mask = mask.reshape(-1, 1, 224, 224)

				diff_u = torch.sum(torch.abs(predictions[0] * mask * 255 - 255 * u_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
				diff_v = torch.sum(torch.abs(predictions[1] * mask * 255 - 255 * v_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
				diff_w = torch.sum(torch.abs(predictions[2] * mask * 255 - 255 * w_map.reshape(-1, 1, 224, 224)), dim=[2, 3]) / torch.sum(mask, dim=[2, 3])
	
				epoch_loss_l1_u_map += torch.sum(diff_u).item()
				epoch_loss_l1_v_map += torch.sum(diff_v).item()
				epoch_loss_l1_w_map += torch.sum(diff_w).item()

			return epoch_loss_l1_u_map / (len(validation_dataset)), epoch_loss_l1_v_map / (len(validation_dataset)), epoch_loss_l1_w_map / (len(validation_dataset))


def main(model, optimizer, training_dataloader, validation_dataloader, loss_function):
	smallest_loss = float("inf")
	for epoch in range(23, NUM_EPOCHS):

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
		torch.save(model.state_dict(), "{}.pt".format(MAIN_DIR_PATH + "CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModelRegressionNewMeshOrientationNewResLayer"))
		print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
	model = AutoencoderPoseEstimationModel()
	model.load_state_dict(torch.load(MAIN_DIR_PATH + "CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModelRegressionNewMeshOrientationNewResLayer.pt.pt", map_location="cpu"))
	model.to(DEVICE)

	optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
	loss_function = nn.CrossEntropyLoss(reduction="sum")
	loss_functions = [nn.L1Loss(reduction="sum"), MultiscaleSsimLossFunction(DEVICE)]
	train_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[0], 
			 				num_images=list(SUBSET_NUM_DATA.values())[0], 
			 				data_augmentation=PoseEstimationAugmentation)
	
	validation_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[1], 
			      				num_images=list(SUBSET_NUM_DATA.values())[1], 
				  				data_augmentation=None)

	training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=32)
	validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=32)

	main(model, optimizer, training_dataloader, validation_dataloader, loss_functions)
