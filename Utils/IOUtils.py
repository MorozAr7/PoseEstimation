import json
import numpy as np


class IOUtils:
	def __init__(self):
		pass

	@staticmethod
	def load_json_file(path: str) -> dict:
		with open(path, "r") as f:
			data_dict = json.load(f)
		f.close()
		return data_dict

	@staticmethod
	def load_numpy_file(path: str) -> np.array:
		with open(path, "rb") as f:
			data_array = np.load(f)
		f.close()
		return data_array

	@staticmethod
	def save_json_file(path: str, data_dict: dict) -> None:
		with open(path, "w") as f:
			json.dump(data_dict, f)
		f.close()

	@staticmethod
	def save_numpy_file(path: str, data_array: np.array) -> None:
		with open(path, 'wb') as f:
			np.save(f, data_array)
		f.close()
