import os
import sys
CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)
import open3d as o3d
from DatasetRenderer.RendererConfig import *
import numpy as np
import random
import cv2
import sys
import json


from Utils.MathUtils import Transformations
from Utils.IOUtils import IOUtils
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
io = IOUtils()
transformations = Transformations()

def sample_pose() -> dict:
    pose_params = {}
    pose_ranges = {"RotX": (-180, 180),
                            "RotY": (-180, 180),
                            "RotZ": (-180, 180),
                            "TransX": (-50, 50),
                            "TransY": (-50, 50),
                            "TransZ": (500, 750)}
    for key, value in pose_ranges.items():
        pose_params[key] = random.randint(value[0], value[1])

    return pose_params

def load_meshes():
    path = "/DatasetRenderer/Models3D/MeshesReconstructed/"
    objects_name = ["Controller", "Servo", "Main", "Motor", "Axle_front", "Battery"]

    meshes = {}
    for name in objects_name:
        print(path + name + "/" + "MeshEdited3.obj")
        mesh = o3d.io.read_triangle_model(path + name + "/" + "MeshEdited3.obj", True)
        print(mesh.MeshInfo.mesh)
        help(mesh.MeshInfo.mesh)
        exit()
        meshes[name] = mesh

    return meshes


def render_image(meshes):
    num_objects = random.randint(4, 7)
    camera_data = io.load_json_file(MAIN_DIR_PATH + "/CameraData/" + CAM_DATA_FILE)
    camera_intrinsic = np.array(camera_data["K"])
    image_h, image_w = camera_data["res_undist"]
    renderer = o3d.visualization.rendering.OffscreenRenderer(image_w, image_h)
    # renderer.scene.set_background(np.array([0, 0, 0, 1]))
    camera_pose = sample_pose()
    camera_pose_T = transformations.get_transformation_matrix_from_pose(camera_pose)


    print(camera_intrinsic, camera_pose_T)
    direction = [0, 0, 0]
    intensity = 1000
    color = [255, 255, 255]

    renderer.scene.scene.set_sun_light(direction, color, intensity)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.scene.enable_light_shadow("directional", True)

    for index in range(num_objects):
        obj_type = random.randint(0, len(meshes.keys()) - 1)
        obj_type = list(meshes.keys())[obj_type]
        mesh = meshes[obj_type]
        mesh.translate([random.randint(50, 150), random.randint(50, 150), random.randint(-15, 15)])
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        mesh_info = o3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo(mesh, str(index), index)

        model = o3d.visualization.rendering.TriangleMeshModel()
        model.MeshInfo = mesh_info
        #model.meshes = [mesh_info]
        #o3d.io.write_triangle_mesh("temp_mesh.obj", mesh)
        #model = o3d.io.read_triangle_model("temp_mesh.obj", True)
        renderer.scene.add_model(f"obj_{index}", model)
        renderer.setup_camera(camera_intrinsic[0:3, 0:3], camera_pose_T, image_w, image_h)
    depth = np.array(renderer.render_to_depth_image())
    image = np.array(renderer.render_to_image())
    del renderer
    cv2.imshow("img", image)
    cv2.waitKey(0)
    cv2.imshow("img", depth)
    cv2.waitKey(0)

meshes = load_meshes()

while True:
    render_image(meshes)








'''import os
import sys
CURR_DIR_PATH = sys.path[0]
MAIN_DIR_PATH = ""
for i in CURR_DIR_PATH.split("/")[:-1]:
    MAIN_DIR_PATH += i + "/"

sys.path.insert(0, MAIN_DIR_PATH)
sys.path.insert(1, CURR_DIR_PATH)
import open3d as o3d
from DatasetRenderer.RendererConfig import *
import numpy as np
import random
import cv2
import sys
import json


from Utils.MathUtils import Transformations
from Utils.IOUtils import IOUtils
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel(0))
io = IOUtils()
transformations = Transformations()

def sample_pose() -> dict:
    pose_params = {}
    pose_ranges = {"RotX": (-180, 180),
                            "RotY": (-180, 180),
                            "RotZ": (-180, 180),
                            "TransX": (0, 0),
                            "TransY": (0, 0),
                            "TransZ": (500, 750)}
    for key, value in pose_ranges.items():
        pose_params[key] = random.randint(value[0], value[1])

    return pose_params

camera_data = io.load_json_file(MAIN_DIR_PATH + "/CameraData/" + CAM_DATA_FILE)
camera_intrinsic = np.array(camera_data["K"])
image_h, image_w = camera_data["res_undist"]
model_controller = o3d.io.read_triangle_model(
    "/Users/artemmoroz/Desktop/CIIRC_projects/PoseEstimation/DatasetRenderer/Models3D/MeshesReconstructed/Controller/MeshEdited3.obj")
model_motor = o3d.io.read_triangle_model("/Users/artemmoroz/Desktop/CIIRC_projects/PoseEstimation/DatasetRenderer/Models3D/MeshesReconstructed/Battery/MeshEdited3.obj", True)
#renderer = o3d.visualization.rendering.OffscreenRenderer(image_w, image_h)
while True:
    renderer = o3d.visualization.rendering.OffscreenRenderer(image_w, image_h)

    renderer.scene.add_model("Name", model_controller)

    sampled_pose1 = sample_pose()
    transformation_matrix1 = transformations.get_transformation_matrix_from_pose(sampled_pose1)

    renderer.setup_camera(camera_intrinsic[0:3, 0:3], transformation_matrix1, image_w, image_h)
    image_no_background1 = np.array(renderer.render_to_image())

    renderer.scene.add_model("Name2", model_motor)
    sampled_pose2 = sample_pose()
    transformation_matrix2 = transformations.get_transformation_matrix_from_pose(sampled_pose2)

    renderer.setup_camera(camera_intrinsic[0:3, 0:3], transformation_matrix2, image_w, image_h)
    a = renderer.render_to_depth_image()
    print(a)
    image_no_background2 = np.array(renderer.render_to_image())
    cv2.imshow("img1", image_no_background1)
    cv2.waitKey(0)
    cv2.imshow("img2", image_no_background2)
    cv2.waitKey(0)

    #del renderer.scene.scene
    #del renderer.scene
    #del renderer"""

'''



