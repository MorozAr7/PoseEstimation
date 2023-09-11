from torch import nn
import torch 

class ProjectionLoss(nn.Module):
    def __init__(self, point_cloud, device):
        super(ProjectionLoss, self).__init__()
        self.L1_loss = nn.L1Loss(reduction="sum")
        
        self.point_cloud = point_cloud
        self.device = device
        self.homogenous_point_cloud = self.get_homogenous_coords(self.point_cloud).T.to(device)
        self.num_points = self.point_cloud.shape[0]

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
    
    def compute_projection_loss(self, T_target, T_coarse_updated):
        transformed_pc_predicted = T_coarse_updated @ self.homogenous_point_cloud
        transformed_pc_target = T_target @ self.homogenous_point_cloud

        return self.L1_loss(transformed_pc_predicted[:, :3, :], transformed_pc_target[:, :3, :])
    
    def forward(self, cnn_translation, cnn_rotation, T_coarse, T_target):
        t_target = T_target[..., :3, -1]
        t_coarse = T_coarse[..., :3, -1]
        R_coarse = T_coarse[..., :3, :3]
        
        updated_z = self.get_updated_depth(cnn_translation[..., 2:], t_coarse)
        updated_xy = self.get_updated_translation(cnn_translation[..., :2], t_coarse, t_target)
        updated_R = self.get_updated_rotation(cnn_rotation, R_coarse)
        
        loss_xy = self.get_xy_loss(updated_xy, T_target)
        loss_z = self.get_z_loss(updated_z, T_target)
        loss_R = self.get_R_loss(updated_R, T_target)
        
        return loss_xy/self.num_points, loss_z/self.num_points, loss_R/self.num_points
        