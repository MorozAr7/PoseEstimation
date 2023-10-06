from CONFIG import *
from CnnModel import FlowEstimationCnn
from LoadDataset import Dataset
from Utils.DataAugmentationUtils import PoseEstimationAugmentation
from torch.utils.data import DataLoader
from LossFunction import MultiscaleSsimLossFunction
import torch.nn as nn
import time
import numpy as np
from warnings import filterwarnings
import cv2
from DatasetRenderer.RenderOpticalFLow import OpticalFlowRenderer

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
    epochs_to_change = list(range(25, 500, 25))
    if epoch in epochs_to_change:
        optimizer.param_groups[0]["lr"] /= 2


def init_classification_model():
    weights_path = MAIN_DIR_PATH + "CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModelRegressionGrayscaleOneDecoder.pt"
    model = FlowEstimationCnn()
    pretrained_dict = torch.load(weights_path, map_location="cpu")

    model_dict = model.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    pretrained_dict_updated = {}
    for key, value in pretrained_dict.items():
        if "Conv8.Conv.weight" in key:
            pretrained_dict_updated[key] = torch.nn.init.xavier_uniform_(torch.rand((4, 64, 3, 3), requires_grad=True))
        else:
            pretrained_dict_updated[key] = value

    model.load_state_dict(pretrained_dict_updated, strict=True)

    return model


def one_epoch(model, optimizer, dataloader, loss_functions, is_training=True, epoch=0):
    model.train() if is_training else model.eval()

    epoch_loss_x = 0
    epoch_loss_y = 0
    epoch_loss_magnitude = 0

    l1_loss_function = loss_functions[0]

    if is_training:
        for index, (image_real, image_rendered, optical_flow, mask) in enumerate(dataloader):
            print("BATCH TRAINING: ", index)
            optimizer.zero_grad()

            image_real = image_real.to(DEVICE)
            image_rendered = image_rendered.to(DEVICE)
            optical_flow = optical_flow.to(DEVICE)
            mask = mask.to(DEVICE)

            predictions = model(image_real, image_rendered)
            #print(mask.shape)
            #print(torch.min(predictions[:, 0:1, ...]), torch.max(predictions[:, 0:1, ...]), torch.min(predictions[:, 1:2, ...]), torch.max(predictions[:, 1:2, ...]), torch.min(predictions[:, 2:3, ...]), torch.max(predictions[:, 2:3, ...]))
            #print(torch.min(optical_flow[:, 0:1, ...]), torch.max(optical_flow[:, 0:1, ...]), torch.min(optical_flow[:, 1:2, ...]), torch.max(optical_flow[:, 1:2, ...]), torch.min(optical_flow[:, 2:3, ...]), torch.max(optical_flow[:, 2:3, ...]))
            loss_x = l1_loss_function(predictions[:, 0, ...] * mask, optical_flow[:, 0, ...] * mask)
            loss_y = l1_loss_function(predictions[:, 1, ...] * mask, optical_flow[:, 1, ...] * mask)
            loss_magnitude = l1_loss_function(predictions[:, 2, ...] * mask, optical_flow[:, 2, ...] * mask)
            total_loss = loss_magnitude + loss_x + loss_y

            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            epoch_loss_x += loss_x.item()
            epoch_loss_magnitude += loss_magnitude.item()
            epoch_loss_y += loss_y.item()
        return epoch_loss_x / len(train_dataset), epoch_loss_magnitude / len(train_dataset), epoch_loss_y / len(train_dataset)
    else:
        with torch.no_grad():
            for index, (image_real, image_rendered, optical_flow, mask) in enumerate(dataloader):
                print("BATCH VALIDATION: ", index)
                optimizer.zero_grad()

                image_real = image_real.to(DEVICE)
                image_rendered = image_rendered.to(DEVICE)
                optical_flow = optical_flow.to(DEVICE)
                mask = mask.to(DEVICE)
                predictions = model(image_real, image_rendered)

                loss_x = l1_loss_function(predictions[:, 0, ...] * mask, optical_flow[:, 0, ...] * mask)
                loss_y = l1_loss_function(predictions[:, 1, ...] * mask, optical_flow[:, 1, ...] * mask)
                loss_magnitude = l1_loss_function(predictions[:, 2, ...] * mask, optical_flow[:, 2, ...] * mask)

                torch.cuda.empty_cache()

                epoch_loss_x += loss_x.item()
                epoch_loss_magnitude += loss_magnitude.item()
                epoch_loss_y += loss_y.item()

        return epoch_loss_x / len(validation_dataset), epoch_loss_magnitude / len(validation_dataset), epoch_loss_y / len(validation_dataset)


def main(model, optimizer, training_dataloader, validation_dataloader, loss_function):
    smallest_loss = float("inf")
    for epoch in range(1, NUM_EPOCHS):

        since = time.time()
        change_learning_rate(optimizer, epoch)
        loss_x_t, loss_m_t, loss_y_t = one_epoch(model,
                                                 optimizer,
                                                 training_dataloader,
                                                 loss_function,
                                                 is_training=True,
                                                 epoch=epoch
                                                 )
        loss_x_v, loss_m_v, loss_y_v = one_epoch(model,
                                                 optimizer,
                                                 validation_dataloader,
                                                 loss_function,
                                                 is_training=False
                                                 )

        print(f"Epoch: {epoch}, ",
              f"Train loss x: {loss_x_t}, ",
              f"Train loss magn: {loss_m_t}, ",
              f"Train loss y: {loss_y_t}, ",
              f"Valid loss x: {loss_x_v}, ",
              f"Valid loss magn: {loss_m_v}, ",
              f"Valid loss y: {loss_y_v}", )

        print("EPOCH RUNTIME", time.time() - since)

        if loss_x_v + loss_m_v + loss_y_v < smallest_loss and SAVE_MODEL:
            smallest_loss = loss_x_v + loss_m_v + loss_y_v
        print("SAVING MODEL")
        torch.save(model.state_dict(), "{}.pt".format(
            MAIN_DIR_PATH + "OpticalFlow/TrainedModels/FlowEstimationEncoder"))
        print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
    model = FlowEstimationCnn().apply(init_weights)
    # model.load_state_dict(torch.load(MAIN_DIR_PATH + "CoarsePoseEstimation/TrainedModels/CoarsePoseEstimatorModelRegressionGrayscaleOneDecoder.pt", map_location="cpu"))
    model.to(DEVICE)
    optical_flow_renderer = OpticalFlowRenderer()
    optimizer = torch.optim.Adam(lr=LEARNING_RATE, params=model.parameters())
    loss_functions = [nn.L1Loss(reduction="sum"), MultiscaleSsimLossFunction(DEVICE), nn.BCELoss(reduction="sum")]
    train_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[0],
                            num_images=list(SUBSET_NUM_DATA.values())[0],
                            dataset_renderer=optical_flow_renderer,
                            data_augmentation=PoseEstimationAugmentation)

    validation_dataset = Dataset(subset=list(SUBSET_NUM_DATA.keys())[1],
                                 num_images=list(SUBSET_NUM_DATA.values())[1],
                                 dataset_renderer=optical_flow_renderer,
                                 data_augmentation=None)

    training_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)  # ,
    # num_workers=32)
    validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)  # ,
    # num_workers=32)

    main(model, optimizer, training_dataloader, validation_dataloader, loss_functions)
