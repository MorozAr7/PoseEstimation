import numpy as np
from CONFIG import *
from torch import nn
import torch 
from Utils.IOUtils import IOUtils
import cv2

class ProjectionLoss(nn.Module):
    def __init__(self, point_cloud, device, projection_type: str = "3D", disentangle: bool = True):
        super(ProjectionLoss, self).__init__()
        self.ProjectionType = projection_type
        self.disentangle = disentangle
        self.L1_loss = nn.L1Loss(reduction="sum")
        self.io = IOUtils()
        self.point_cloud = point_cloud
        self.device = device
        self.homogenous_point_cloud = self.get_homogenous_coords(self.point_cloud).T.to(device)
        self.num_points = self.point_cloud.shape[0]
        self.camera_data = self.io.load_json_file(MAIN_DIR_PATH + "/CameraData/camera_data_1.json")
        self.height, self.width = self.camera_data["res_undist"]
        self.camera_intrinsic = torch.tensor(self.camera_data["K"]).to(self.device)

    def get_homogenous_coords(self, point_cloud):
        ones = torch.ones(size=(point_cloud.shape[0], 1)).to(self.device)
        return torch.cat([point_cloud, ones], dim=-1)
    
    @staticmethod
    def get_updated_translation(xy_prediction, t_coarse, t_target):
        return (xy_prediction + t_coarse[..., :2] / t_coarse[..., 2:3]) * t_target[..., 2:3]
    
    @staticmethod
    def get_updated_depth(z_prediction, t_coarse):
        return z_prediction * t_coarse[..., 2:3]
    
    @staticmethod
    def get_updated_rotation(R_prediction, R_coarse):
        return torch.bmm(R_prediction, R_coarse)
    
    def get_xy_loss(self, updated_xy, T_target):
        T_updated = T_target.clone()
        T_updated[..., :2, -1] = updated_xy
        return self.compute_projection_loss(T_target, T_updated)
    
    def get_z_loss(self, updated_z, T_target):
        T_updated = T_target.clone()
        T_updated[..., 2:3, -1] = updated_z
        return self.compute_projection_loss(T_target, T_updated)
    
    def get_R_loss(self, updated_R, T_target):
        T_updated = T_target.clone()
        T_updated[..., :3, :3] = updated_R
        return self.compute_projection_loss(T_target, T_updated)
    
    def get_total_loss(self, updated_xy, updated_z, updated_R, T_target):
        T_updated = T_target.clone()
        T_updated[..., :3, :3] = updated_R
        T_updated[..., :2, -1] = updated_xy
        T_updated[..., 2:3, -1] = updated_z
        return self.compute_projection_loss(T_target, T_updated, vis=False)
    
    
    def visualize_projection(self, xy_projected_prediction, xy_projected_target):
        xy_projected_prediction = torch.tensor(xy_projected_prediction, dtype=torch.int)[0].permute(1, 0).detach().cpu().numpy()
        xy_projected_target = torch.tensor(xy_projected_target, dtype=torch.int)[0].permute(1, 0).detach().cpu().numpy()
        
        y_prediction = np.clip(xy_projected_prediction[:, 1], 0, self.height - 1)
        x_prediction = np.clip(xy_projected_prediction[:, 0], 0, self.width - 1)
        y_target = np.clip(xy_projected_target[:, 1], 0, self.height - 1)
        x_target = np.clip(xy_projected_target[:, 0], 0, self.width - 1)
        
        #print(np.max(xy_projected_target), np.min(xy_projected_target))
        frame_prediction = np.zeros(shape=(self.height, self.width))
        frame_target = np.zeros(shape=(self.height, self.width))
        
        frame_prediction[y_prediction, x_prediction] = 1
        frame_target[y_target, x_target] = 1
        #print(np.stack([frame_prediction, frame_target], axis=1).shape)
        visualize = np.vstack([frame_prediction, frame_target]) * 255
        #print(np.max(visualize), np.min(visualize))
        cv2.imshow("result", visualize)
        cv2.waitKey(0)
        
        
    def compute_projection_loss(self, T_target, T_coarse_updated, vis=False):
        transformed_pc_prediction = T_coarse_updated @ self.homogenous_point_cloud
        transformed_pc_target = T_target @ self.homogenous_point_cloud
 
        if self.ProjectionType == "3D":
            return self.L1_loss(transformed_pc_prediction[:, :3, :], transformed_pc_target[:, :3, :])
        elif self.ProjectionType == "2D":
            transformed_camera_pc_prediction = self.camera_intrinsic @ transformed_pc_prediction
            transformed_camera_pc_target = self.camera_intrinsic @ transformed_pc_target
            
            transformed_camera_pc_prediction[:, 2:3, :][torch.where(transformed_pc_prediction[:, 2:3, :] == 0)] = 1
            transformed_camera_pc_target[:, 2:3, :][torch.where(transformed_pc_target[:, 2:3, :] == 0)] = 1
      
            xy_projected_predicted = transformed_camera_pc_prediction[:, :2, :] / (transformed_camera_pc_prediction[:, 2:3, :])
            xy_projected_target = transformed_camera_pc_target[:, :2, :] / (transformed_camera_pc_target[:, 2:3, :])
            
            if vis:
                self.visualize_projection(xy_projected_predicted, xy_projected_target)
            return self.L1_loss(xy_projected_predicted, xy_projected_target)
        else:
            exit("Unknown projection type!")
    
    def forward(self, cnn_translation, cnn_rotation, T_coarse, T_target):
        t_target = T_target[..., :3, -1]
        t_coarse = T_coarse[..., :3, -1]
        R_coarse = T_coarse[..., :3, :3]
        
        updated_z = self.get_updated_depth(cnn_translation[..., 2:], t_coarse)
        updated_xy = self.get_updated_translation(cnn_translation[..., :2], t_coarse, t_target)
        updated_R = self.get_updated_rotation(cnn_rotation, R_coarse)

        loss_xy = self.get_xy_loss(updated_xy, T_target)/self.num_points
        loss_z = self.get_z_loss(updated_z, T_target)/self.num_points
        loss_R = self.get_R_loss(updated_R, T_target)/self.num_points
        loss_projection = self.get_total_loss(updated_xy, updated_z, updated_R, T_target)/self.num_points
        loss_data = {"LossR": loss_R, "LossXY": loss_xy, "LossZ": loss_z, "LossTotal": loss_projection}
    
        return loss_data