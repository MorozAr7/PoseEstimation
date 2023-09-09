import numpy as np


class RotationConversions:
	def __init__(self):
		pass

	@staticmethod
	def matrix2quaternion(rot_matrix):
		assert rot_matrix.shape == (3, 3) and abs(np.linalg.det(rot_matrix) - 1.0) < 1e-6
		trace = np.matrix.trace(rot_matrix)
		if trace > 0:
			S = 2 * np.sqrt(trace + 1)
			qw = 0.25 * S
			qx = (rot_matrix[2][1] - rot_matrix[1][2]) / S
			qy = (rot_matrix[0][2] - rot_matrix[2][0]) / S
			qz = (rot_matrix[1][0] - rot_matrix[0][1]) / S
		elif rot_matrix[0][0] > rot_matrix[1][1] and rot_matrix[0][0] > rot_matrix[2][2]:
			S = 2 * np.sqrt(1 + rot_matrix[0][0] - rot_matrix[1][1] - rot_matrix[2][2])
			qw = (rot_matrix[2][1] - rot_matrix[1][2]) / S
			qx = 0.25 * S
			qy = (rot_matrix[0][1] + rot_matrix[1][0]) / S
			qz = (rot_matrix[0][2] + rot_matrix[2][0]) / S
		elif rot_matrix[1][1] > rot_matrix[2][2]:
			S = 2 * np.sqrt(1 + rot_matrix[1][1] - rot_matrix[0][0] - rot_matrix[2][2])
			qw = (rot_matrix[0][2] - rot_matrix[2][0]) / S
			qx = (rot_matrix[0][1] + rot_matrix[1][0]) / S
			qy = 0.25 * S
			qz = (rot_matrix[1][2] + rot_matrix[2][1]) / S
		else:
			S = 2 * np.sqrt(1 + rot_matrix[2][2] - rot_matrix[0][0] - rot_matrix[1][1])
			qw = (rot_matrix[1][0] - rot_matrix[0][1]) / S
			qx = (rot_matrix[0][2] + rot_matrix[2][0]) / S
			qy = (rot_matrix[1][2] + rot_matrix[2][1]) / S
			qz = 0.25 * S

		return np.array([qw, qx, qy, qz])

	@staticmethod
	def quaternion2matrix(quaternion):
		assert (np.sqrt(np.sum(np.square(quaternion))) - 1) < 1e-6
		rot_matrix = np.zeros(shape=(3, 3))
		square_q = np.square(quaternion)
		rot_matrix[0][0] = square_q[0] + square_q[1] - square_q[2] - square_q[3]
		rot_matrix[1][1] = square_q[0] - square_q[1] + square_q[2] - square_q[3]
		rot_matrix[2][2] = square_q[0] - square_q[1] - square_q[2] + square_q[3]

		product1 = quaternion[1] * quaternion[2]
		product2 = quaternion[0] * quaternion[3]
		rot_matrix[1][0] = 2 * (product1 + product2)
		rot_matrix[0][1] = 2 * (product1 - product2)

		product1 = quaternion[1] * quaternion[3]
		product2 = quaternion[0] * quaternion[2]
		rot_matrix[2][0] = 2 * (product1 - product2)
		rot_matrix[0][2] = 2 * (product1 + product2)

		product1 = quaternion[2] * quaternion[3]
		product2 = quaternion[0] * quaternion[1]
		rot_matrix[2][1] = 2 * (product1 + product2)
		rot_matrix[1][2] = 2 * (product1 - product2)

		return rot_matrix

	@staticmethod
	def axis_angle2quaternion(axis_angle):
		angle_deg = np.sqrt(np.sum(np.square(axis_angle)))
		angle_rad = np.deg2rad(angle_deg)

		axis_angle_normalized = axis_angle / angle_deg
		sinus = np.sin(angle_rad / 2)
		qw = np.cos(angle_rad / 2)
		qx = axis_angle_normalized[1] * sinus
		qy = axis_angle_normalized[2] * sinus
		qz = axis_angle_normalized[3] * sinus

		return np.array([qw, qx, qy, qz])


class Transformations:
	def __init__(self):
		pass
	
	@staticmethod
	def vector_cross_product(v1, v2):
		pass

	@staticmethod
	def gramm_schmidd_orthogonalization(v1, v2):
		pass

	@staticmethod
	def convert_deg2rad(angle: float = 0.0) -> float:
		return np.deg2rad(angle).item()

	@staticmethod
	def get_rotation_matrix_x(angle: float = 0.0) -> np.array:
		Rx = np.array([[1, 0, 0, 0],
		               [0, np.cos(angle), -np.sin(angle), 0],
		               [0, np.sin(angle), np.cos(angle), 0],
		               [0, 0, 0, 1]])

		return Rx

	@staticmethod
	def get_rotation_matrix_y(angle: float = 0.0) -> np.array:
		Ry = np.array([[np.cos(angle), 0, np.sin(angle), 0],
		               [0, 1, 0, 0],
		               [-np.sin(angle), 0, np.cos(angle), 0],
		               [0, 0, 0, 1]])

		return Ry

	@staticmethod
	def get_rotation_matrix_z(angle: float = 0.0) -> np.array:
		Rz = np.array([[np.cos(angle), -np.sin(angle), 0, 0],
		               [np.sin(angle), np.cos(angle), 0, 0],
		               [0, 0, 1, 0],
		               [0, 0, 0, 1]])

		return Rz

	@staticmethod
	def get_translation_matrix(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> np.array:
		translation_matrix = np.zeros(shape=(4, 4))
		translation_matrix[0:3, -1] = [x, y, z]

		return translation_matrix

	def get_object_rotation_matrix(self, angle_x: float = 0.0, angle_y: float = 0.0, angle_z: float = 0.0) -> np.array:
		angle_x = self.convert_deg2rad(angle_x)
		angle_y = self.convert_deg2rad(angle_y)
		angle_z = self.convert_deg2rad(angle_z)

		return self.get_rotation_matrix_z(angle_z) @ (self.get_rotation_matrix_y(angle_y) @ self.get_rotation_matrix_x(angle_x))

	def get_transformation_matrix_from_pose(self, transformation_params: dict) -> np.array:
		rotation_matrix = self.get_object_rotation_matrix(angle_x=transformation_params["RotX"],
		                                                  angle_y=transformation_params["RotY"],
		                                                  angle_z=transformation_params["RotZ"])

		translation_matrix = self.get_translation_matrix(x=transformation_params["TransX"],
		                                                 y=transformation_params["TransY"],
		                                                 z=transformation_params["TransZ"])

		assert translation_matrix.shape == (4, 4)
		assert rotation_matrix.shape == (4, 4)

		return rotation_matrix + translation_matrix
