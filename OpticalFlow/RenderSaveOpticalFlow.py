import os
from CONFIG import *
from DatasetRenderer.RenderOpticalFLow import OpticalFlowRenderer
from Utils.IOUtils import IOUtils
import cv2
from Utils.MathUtils import Transformations


def crop_and_resize(array, bbox_corner):
    x_min, y_min, x_max, y_max = bbox_corner

    cropped = array[y_min:y_max, x_min:x_max]
    try:
        resized = cv2.resize(cropped, (224, 224))
        return resized
    except Exception as e:
        print(e)
        print(x_min, x_max, y_min, y_max)
        exit()


def render_and_save(subset, index_real, optical_flow_renderer, io_utils, transformation_utils):
    path = MAIN_DIR_PATH + "Dataset/" + subset + "/"
    json_data = io_utils.load_json_file(path + "Pose/" + "data_{}.json".format(index_real))
    pose_real = json_data["Pose"]
    trans_matrix_real = transformation_utils.get_transformation_matrix_from_pose(pose_real)
    path_datapoint = path + "ImageRefinement/" + f"Data_{index_real}/"
    path_flow = path_datapoint + "Flow/"
    try:
        os.mkdir(path_flow)
    except FileExistsError:
        print('The directory already exists, continue writing to it')
    for index_rendered in range(10):
        rendered_json_data_path = path_datapoint + "Pose/" + f"data_{index_rendered}.json"
        rendered_data = io_utils.load_json_file(rendered_json_data_path)
        rendered_pose = rendered_data["Pose"]
        rendered_box = rendered_data["Box"]
        trans_matrix_rendered = transformation_utils.get_transformation_matrix_from_pose(rendered_pose)
        optical_flow_map = optical_flow_renderer.get_optical_flow(trans_matrix_real, trans_matrix_rendered).reshape(1039, 1865, 3)
        reshaped_flow = crop_and_resize(optical_flow_map, rendered_box)
        io_utils.save_numpy_file(path_flow + f"data_{index_rendered}.npy", reshaped_flow)

def main():
    data_num_dict = {"Training": 256, "Validation": 1}
    optic_flow_renderer = OpticalFlowRenderer()
    io = IOUtils()
    transformations = Transformations()
    for subset in ["Training", "Validation"]:
        for index in range(data_num_dict[subset]):
            print(subset, index)
            render_and_save(subset, index, optic_flow_renderer, io, transformations)


if __name__ == "__main__":
    main()
