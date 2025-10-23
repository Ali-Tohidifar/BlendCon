import math
import random
import yaml
import pickle
import os
import logging
import os
import zoneinfo
import bpy
import bpy_extras
from pathlib import Path
from mathutils import Vector
from datetime import datetime, timezone

# Your current timestamp in UTC
timestamp_utc = datetime.now(timezone.utc)
# Define the EDT timezone
edt_timezone = zoneinfo.ZoneInfo('America/New_York')
# Convert the timestamp to EDT
timestamp_edt = timestamp_utc.astimezone(edt_timezone)
timestamp = timestamp_edt.strftime('%y-%m-%d-%H-%M')


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataGenerator')

# Ensure log messages are saved to a file
log_file = f'/workspace/logs/{timestamp}_data_generation.log'
if not os.path.exists(os.path.dirname(log_file)):
    os.makedirs(os.path.dirname(log_file))
fh = logging.FileHandler(log_file)
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)
logger.info('Loading configuration file.')

logger.info('Starting data generation process.')

Code_dir = Path(os.getcwd())

"Read config and user inputs"
# Read config yamlfile
# os.chdir(Code_dir)
with open(Code_dir / "config.yaml", "r") as yamlfile:
    config = yaml.load(yamlfile, Loader=yaml.FullLoader)

# Setup directories
Dataset_dir = Code_dir / 'Dataset'
Avatar_dir = Code_dir / 'Avatars'
Scene_dir = Code_dir / 'Scenes'
# report_path = Code_dir / 'Report.txt'
horizon_path = Code_dir / 'Horizon.blend'
empty_dir = Code_dir / 'Empty.blend'

# Number of workers
Number_of_Workers = int(config['Number_of_Workers'])

# Camera prameters
Camera_Radius = int(random.gauss(int(config['Camera_Radius']), int(config['Camera_Radius'])/3))
if Camera_Radius < 2: Camera_Radius = 2
Iterations_Avatar_Location_Randomization = int(config['Iterations_Avatar_Location_Randomization'])
Iterations_Lighting_Randomization = int(config['Iterations_Lighting_Randomization'])
Number_of_Image_Sequences =  int(config['Number_of_Image_Sequences'])
Drone_View = config['Drone_View']

# Avatars path
avatar_paths = [Avatar_dir / avatar for avatar in os.listdir(Avatar_dir) if '.blend' in avatar]

# Scene path
scene_paths = [Scene_dir / scene for scene in os.listdir(Scene_dir) if '.blend' in scene]

# Initialize the scene
scene = bpy.context.scene
camera = bpy.context.scene.camera



"""
Setup GPU in Blender
"""
logger.info('Setting up devices in Blender.')
logger.info('Prefs for Cycles is {}'.format(bpy.context.preferences.addons['cycles'].preferences.__str__()))
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
devices = prefs.get_devices_for_type('CUDA')
logger.info(f'Available devices are {prefs.get_devices_for_type("CUDA").__str__()}.')
for device in devices:
    device.use = True # enable any available GPU and CPU
    logger.info(f'{device["name"]}, use: {device["use"]}')    
    

"""
Functions
"""

"New Camera Function"

def new_camera(focal_len=20):
    # Removing the camera
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.data.objects['Camera'].select_set(True)
        if bpy.data.objects['Camera'].select_get():
            bpy.ops.object.delete()
    except:
        pass

    # create the first camera data
    bpy.ops.object.camera_add(location=(0, 0, 0),
                              rotation=(0, 0, 0))
    scene.camera = bpy.context.object
    scene.camera.data.lens = focal_len
    camera = bpy.context.scene.camera


"Oclussion detector for workers with bones"

def occlusion_detector(target_arm, origin):
    # collect all other mesh objects that can occlude target
    others = [obj for obj in bpy.data.objects if (
        obj.parent != target_arm and obj != origin and obj.type == 'MESH' and 'cam_circle' not in obj.name)]

    # add cubes in bone locations
    added_cubes = []
    for bone in target_arm.pose.bones:
        bonePos = target_arm.matrix_world @ bone.head
        bpy.ops.mesh.primitive_cube_add(
            size=0.02, enter_editmode=False, align='WORLD', location=bonePos, scale=(1, 1, 1))
        added_cubes.append(bpy.context.active_object)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # iterate through target cubes and identify occlusion
    occlusion = 0
    for target in added_cubes:
        # calculate target hit distance
        target_in_target_space = target.matrix_world.inverted() @ target.location
        origin_in_target_space = target.matrix_world.inverted() @ origin.location
        ray_direction = target_in_target_space - origin_in_target_space

        ray_cast_target = target.ray_cast(
            origin_in_target_space, ray_direction)
        hit_loc_in_glob = target.matrix_world @ ray_cast_target[1]
        target_distance = hit_loc_in_glob - origin.location

        # loop through others and detect occlusion
        for obj in others:
            # calculate non-target hit distance
            origin_in_obj_space = obj.matrix_world.inverted() @ origin.location
            target_in_obj_space = obj.matrix_world.inverted() @ target.location
            ray_direction = target_in_obj_space - origin_in_obj_space

            ray_cast_obj = obj.ray_cast(origin_in_obj_space, ray_direction)
            hit_loc_in_glob = obj.matrix_world @ ray_cast_obj[1]
            obj_distance = hit_loc_in_glob - origin.location

            if ray_cast_obj[0] and target_distance.length > obj_distance.length:
                occlusion += 1

    # calculate occlusion percentage
    precentage = occlusion/len(added_cubes)

    # remove added cubes
    for cube in added_cubes:
        cube.select_set(True)
    bpy.ops.object.delete()

    return precentage


"Joint Tracker Function"

def joint_tracker(lighting, workers_name_list, path=Dataset_dir):
    #TODO
    workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
    ###

    # Create a dictionary for data capture
    Information_dict = {}

    # Capturing the bone names and connections of workers once and use it in each frame iteration
    bone_connection_dict = {}
    bones_list_dict = {}
    leaf_bones_dict = {}
    root_bones_dict = {}

    for worker in workers:
        # find root and leaf bones
        bones_list = []
        leaf_bones = []
        root_bones = []

        for bone in worker.pose.bones:
            bones_list.append(bone.name)
            if len(bone.children) == 0:
                leaf_bones.append(bone)
            elif bone.parent is None:
                root_bones.append(bone)
        bones_list_dict[str(worker.name)] = bones_list
        leaf_bones_dict[str(worker.name)] = [bone.name for bone in leaf_bones]
        root_bones_dict[str(worker.name)] = [bone.name for bone in root_bones]

        # start from leaf bone and iterate through parents and append connections
        sklet_set = set()    # using set here to aviod duplications
        for bone in leaf_bones:
            iter_bone = bone
            while iter_bone.parent != None:
                sklet_set.add((iter_bone.name, iter_bone.parent.name))
                iter_bone = iter_bone.parent
        bone_connection_dict[str(worker.name)] = list(sklet_set)
    logger.info('Ground truths are being extracted.')
    # print('Ground truths are being extracted:\n')
    total = bpy.context.scene.frame_end+1 - bpy.context.scene.frame_start
    # Looping over the animation frames
    for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end+1, bpy.context.scene.frame_step):
        # print progress precentage
        progress = int(frame/total*100)
        progress_report = '%' + str(progress)
        # print(progress_report,end='')
        logger.info(progress_report)
        
        # setting scene to the required frame
        bpy.context.scene.frame_set(frame)

        # define an intermidate dictionaries for each of the frames
        info_on_each_frame = {}

        # calculate render size
        render_scale = bpy.context.scene.render.resolution_percentage / 100
        render_size = (int(bpy.context.scene.render.resolution_x * render_scale),
                       int(bpy.context.scene.render.resolution_y * render_scale))

        # save render size in dictionary
        info_on_each_frame['render_size'] = render_size

        # save camera location
        info_on_each_frame['camera_location'] = list(
            bpy.context.scene.camera.location)

        # loop over workers
        for worker in workers:
            # creating worker name tag
            rig_name = worker.name.replace('Armature: ', '')

            # define an intermidate dictionaries for each worker in each frames
            info_on_each_frame_for_each_worker = {}

            # calculate occlusion
            occlusion_percentage = occlusion_detector(
                worker, bpy.context.scene.camera)
            info_on_each_frame_for_each_worker['occlusion'] = occlusion_percentage

            # add bone names to info dict
            info_on_each_frame_for_each_worker['bone_name'] = bones_list_dict[str(
                worker.name)]

            # add root_bones to info dict
            info_on_each_frame_for_each_worker['root_bones'] = root_bones_dict[str(
                worker.name)]

            # add leaf_bones to info dict
            info_on_each_frame_for_each_worker['leaf_bones'] = leaf_bones_dict[str(
                worker.name)]

            # add bone_connection to info dict
            info_on_each_frame_for_each_worker['bone_connection'] = bone_connection_dict[str(
                worker.name)]

            # setting the intermidate dictionaries for each of the frames
            worker_bone_3d_location_dict = {}
            worker_bone_2d_location_dict = {}
            worker_bone_pixel_location_dict = {}

            # initializing the bounding boxes max and min coordinations
            bounding_box_max_x = -1*float('inf')
            bounding_box_max_y = -1*float('inf')
            bounding_box_min_x = 1*float('inf')
            bounding_box_min_y = 1*float('inf')

            # initializing the 3D bounding boxes max and min coordinations
            bounding_box_3D_max_x = -1*float('inf')
            bounding_box_3D_max_y = -1*float('inf')
            bounding_box_3D_max_z = -1*float('inf')
            bounding_box_3D_min_x = 1*float('inf')
            bounding_box_3D_min_y = 1*float('inf')
            bounding_box_3D_min_z = 1*float('inf')

            # looping over each bone
            for bone in worker.data.bones:
                # Selecting the bones. This can be head, tail, center...
                # More info: https://docs.blender.org/api/current/bpy.types.PoseBone.html?highlight=bpy%20bone#bpy.types.PoseBone.bone
                bone_pos = worker.pose.bones[bone.name].head

                # Converting bone local position to the scene coordinate
                # Since Blender 2.8 you multiply matrices with @ not with *
                bone_pos_glob = worker.matrix_world @ bone_pos

                #  create bone_location_3d dict
                worker_bone_3d_location_dict[str(bone.name)] = [
                    bone_pos_glob[0], bone_pos_glob[1], bone_pos_glob[2]]

                # Converting 3d location of iterating bone to pixel coordinates
                coordinate_2d = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, bone_pos_glob)

                #  create bone_location_3d dict
                worker_bone_2d_location_dict[str(bone.name)] = [
                    coordinate_2d[0], coordinate_2d[1], coordinate_2d[2]]

                # Filtering the pixel coordinates behind the camera
                if coordinate_2d.z < 0:
                    bounding_box_min_x = 1*float('inf')
                    bounding_box_max_x = -1*float('inf')
                    bounding_box_min_y = 1*float('inf')
                    bounding_box_max_y = -1*float('inf')

                    bounding_box_3D_max_x = -1*float('inf')
                    bounding_box_3D_max_y = -1*float('inf')
                    bounding_box_3D_max_z = -1*float('inf')
                    bounding_box_3D_min_x = 1*float('inf')
                    bounding_box_3D_min_y = 1*float('inf')
                    bounding_box_3D_min_z = 1*float('inf')

                else:
                    pixel_coordinate_x = coordinate_2d.x * render_size[0]
                    pixel_coordinate_y = render_size[1] - \
                        (coordinate_2d.y * render_size[1])

                    # Saving bone locations of each workers
                    worker_bone_pixel_location_dict[str(bone.name)] = [
                        pixel_coordinate_x, pixel_coordinate_y, coordinate_2d.z]

                    ### Fix this part ###
                    # Selecting the max and min bounding box coordinates
                    if pixel_coordinate_x < bounding_box_min_x:
                        bounding_box_min_x = pixel_coordinate_x
                    if pixel_coordinate_x > bounding_box_max_x:
                        bounding_box_max_x = pixel_coordinate_x
                    if pixel_coordinate_y < bounding_box_min_y:
                        bounding_box_min_y = pixel_coordinate_y
                    if pixel_coordinate_y > bounding_box_max_y:
                        bounding_box_max_y = pixel_coordinate_y

                    # Selecting the max and min 3D bounding box coordinates
                    if bone_pos[0] < bounding_box_3D_min_x:
                        bounding_box_3D_min_x = bone_pos[0]
                    if bone_pos[0] > bounding_box_3D_max_x:
                        bounding_box_3D_max_x = bone_pos[0]
                    if bone_pos[1] < bounding_box_3D_min_y:
                        bounding_box_3D_min_y = bone_pos[1]
                    if bone_pos[1] > bounding_box_3D_max_y:
                        bounding_box_3D_max_y = bone_pos[1]
                    if bone_pos[2] < bounding_box_3D_min_z:
                        bounding_box_3D_min_z = bone_pos[2]
                    if bone_pos[2] > bounding_box_3D_max_z:
                        bounding_box_3D_max_z = bone_pos[2]
                    ###

                # iterate through meshes and extract 3d optimas and pixel optimas for each workers

                # clamp the bounding boxes to the render size

            # writing the data to the dictionaries
            info_on_each_frame_for_each_worker['bone_location_2d_raw'] = worker_bone_2d_location_dict

            if bounding_box_min_x == 1*float('inf') or bounding_box_max_x == -1*float('inf') or bounding_box_min_y == 1*float('inf') or bounding_box_max_y == -1*float('inf') or bounding_box_3D_min_x == 1*float('inf') or bounding_box_3D_max_x == -1*float('inf') or bounding_box_3D_min_y == 1*float('inf') or bounding_box_3D_max_y == -1*float('inf') or bounding_box_3D_min_z == 1*float('inf') or bounding_box_3D_max_z == -1*float('inf'):
                continue

            else:
                # add bone info and bb info to info dict
                info_on_each_frame_for_each_worker['BB2D'] = [
                    [bounding_box_min_x, bounding_box_min_y], [bounding_box_max_x, bounding_box_max_y]]
                info_on_each_frame_for_each_worker['BB3D_global_coordinate'] = [[bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_min_z], [
                    bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_max_z]]
                info_on_each_frame_for_each_worker['bone_location_3d'] = worker_bone_3d_location_dict
                info_on_each_frame_for_each_worker['bone_location_2d'] = worker_bone_pixel_location_dict

                # Convert 3D BB in global coordinate to pixel coordinate
                p1_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_min_z))
                p2_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_min_y, bounding_box_3D_min_z))
                p3_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_min_z))
                p4_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_max_y, bounding_box_3D_min_z))

                p6_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_min_y, bounding_box_3D_max_z))
                p7_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_min_y, bounding_box_3D_max_z))
                p8_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_max_x, bounding_box_3D_max_y, bounding_box_3D_max_z))
                p5_3DBB_glob = worker.matrix_world @ Vector(
                    (bounding_box_3D_min_x, bounding_box_3D_max_y, bounding_box_3D_max_z))

                p1_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p1_3DBB_glob)
                p2_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p2_3DBB_glob)
                p3_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p3_3DBB_glob)
                p4_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p4_3DBB_glob)
                p5_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p5_3DBB_glob)
                p6_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p6_3DBB_glob)
                p7_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p7_3DBB_glob)
                p8_3DBB_camera = bpy_extras.object_utils.world_to_camera_view(
                    bpy.context.scene, bpy.context.scene.camera, p8_3DBB_glob)

                p1_3DBB_pixel = [p1_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p1_3DBB_camera.y), p1_3DBB_camera.z]
                p2_3DBB_pixel = [p2_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p2_3DBB_camera.y), p2_3DBB_camera.z]
                p3_3DBB_pixel = [p3_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p3_3DBB_camera.y), p3_3DBB_camera.z]
                p4_3DBB_pixel = [p4_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p4_3DBB_camera.y), p4_3DBB_camera.z]
                p5_3DBB_pixel = [p5_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p5_3DBB_camera.y), p5_3DBB_camera.z]
                p6_3DBB_pixel = [p6_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p6_3DBB_camera.y), p6_3DBB_camera.z]
                p7_3DBB_pixel = [p7_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p7_3DBB_camera.y), p7_3DBB_camera.z]
                p8_3DBB_pixel = [p8_3DBB_camera.x * render_size[0],
                                 render_size[1] * (1 - p8_3DBB_camera.y), p8_3DBB_camera.z]

                # save 3D bounding box's edges as tuples of corners
                info_on_each_frame_for_each_worker['BB3D'] = [(p1_3DBB_pixel, p2_3DBB_pixel), (p2_3DBB_pixel, p3_3DBB_pixel), (p3_3DBB_pixel, p4_3DBB_pixel), (p4_3DBB_pixel, p5_3DBB_pixel), (p4_3DBB_pixel, p1_3DBB_pixel), (
                    p5_3DBB_pixel, p6_3DBB_pixel), (p6_3DBB_pixel, p7_3DBB_pixel), (p6_3DBB_pixel, p1_3DBB_pixel), (p7_3DBB_pixel, p8_3DBB_pixel), (p7_3DBB_pixel, p2_3DBB_pixel), (p8_3DBB_pixel, p3_3DBB_pixel), (p8_3DBB_pixel, p5_3DBB_pixel)]

            # writing the data to the dictionaries
            info_on_each_frame[rig_name] = info_on_each_frame_for_each_worker

        Information_dict[str(frame)] = info_on_each_frame
        Information_dict['lighting'] = lighting
        Information_dict['workers_name_list'] = workers_name_list

    # # Setting the directory for saving dictionaries
    # os.chdir(path)
    logger.debug(f'Saving pickle to {path + "/Joint_Tracker.pickle"}')
    # Saving dictionaries
    with open('Joint_Tracker.pickle', 'wb') as handle:
        pickle.dump(Information_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)



"Depth Map Function"
def Depth_Map_Genrator(output_path):
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["View Layer"].use_pass_mist = True
    bpy.context.scene.world.mist_settings.start = 0
    bpy.context.scene.world.mist_settings.depth = 25

    tree = bpy.context.scene.node_tree
    links = tree.links

    # clear the previously created nodes
    for node in tree.nodes:
        if node.name == "Depth_Maper" or node.name == "Invertor" or node.name == "Depth_Map_Output":
            tree.nodes.remove(node)

    rl = tree.nodes['Render Layers']

    # creating and adjusting Output node's settings
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    output.name = "Depth_Map_Output"
    output.base_path = output_path
    output.format.color_mode = 'BW'
    #FIXME: set the color depth to 16
    # import ipdb; ipdb.set_trace()
    # output.format.color_depth = '16'
    output.location = [420, 230]

    # linking the render layer with output
    links.new(rl.outputs['Mist'], output.inputs['Image'])


"Segmantation function"
def Setup_Segmentation(output_path, occlusion_thrsh=0.98):

    bpy.context.scene.view_layers["View Layer"].use_pass_object_index = True

    workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
    workers = sorted(workers, key=lambda x: x.name)     # sort the workers based on their name to make masks consistant
    
    index = 254
    # indexing the worker object in scene
    for worker in workers:
        for obj in bpy.data.objects:
            if (obj.parent == worker and obj.type == 'MESH'):
                obj.pass_index = index
        index -= 10
    
    background_index = 50
    
    for obj in bpy.data.objects:
        if obj.type =='MESH':
            if all(term not in obj.name for term in ['Floor', 'Horizon', 'Amrature']):
                if obj.parent:
                    if 'Armature' not in obj.parent.name:
                        obj.pass_index = background_index
                else:
                    obj.pass_index = background_index
            elif 'Horizon' in obj.name:
                obj.pass_index = 0 
            else:
                obj.pass_index = background_index

    # creating the required nodes for segmentation
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # clear the previously created nodes
    for node in tree.nodes:
        if node.name == "Segmentation_Calculator" or node.name == 'Segmentation_Output':
            tree.nodes.remove(node)

    # definition of Segmentation_Calculator
    convertor = tree.nodes.new(type="CompositorNodeMath")
    convertor.name = "Segmentation_Calculator"
    convertor.operation = 'DIVIDE'
    convertor.inputs[1].default_value = 255
    convertor.location = [220, 0]

    # linking the Segmentation_Calculator node to the Render Layers
    links = tree.links
    link = links.new(
        tree.nodes['Render Layers'].outputs['IndexOB'], convertor.inputs[0])

    # creating and adjusting Output node's settings
    output = tree.nodes.new(type="CompositorNodeOutputFile")
    output.name = 'Segmentation_Output'
    output.base_path = output_path
    output.format.color_mode = 'BW'
    output.location = [420, 0]

    # linking the Ouput node to the Render Layers and Math
    link2 = links.new(convertor.outputs['Value'], output.inputs['Image'])
    
    

"Render Engine Settings"
def render_setting(scene=bpy.context.scene):
    bpy.context.scene.cycles.max_bounces = int(config['max_bounces'])
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.samples = int(config['samples'])
    bpy.context.scene.cycles.time_limit = 1
    bpy.context.scene.cycles.tile_size = int(config['tile_size'])
    bpy.context.scene.cycles.adaptive_threshold = float(config['adaptive_threshold'])
    bpy.context.scene.render.resolution_x = int(config['resolution_x'])
    bpy.context.scene.render.resolution_y = int(config['resolution_y'])
    bpy.context.scene.render.fps = int(config['Framerate'])
    bpy.context.scene.frame_step = int(config['Framerate'])
    bpy.context.scene.render.image_settings.file_format = 'JPEG'


"Rendering Function"


def rendering_random_camera(lighting, itr, camera, target_rig, scene_name, workers_name_list, name_tag):
    # track target rig by camera
    camera.constraints.new(type="TRACK_TO")
    camera.constraints['Track To'].target = target_rig
    camera.constraints['Track To'].subtarget = 'mixamorig:Spine'

    # remove previous cam_circles from scene
    for ob in bpy.data.objects.values():
        ob.select_set(False)
        try:
            for object in bpy.data.objects:
                if 'cam_circle' in object.name:
                    object.select_set(True)
                    if object.select_get():
                        bpy.ops.object.delete(use_global=False, confirm=False)
        except:
            pass

    # add cam_circle to scene
    cam_circle_radius = Camera_Radius
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=cam_circle_radius,
        enter_editmode=False,
        align='WORLD',
        location=(target_rig.location.x, target_rig.location.y,
                  target_rig.location.z + cam_circle_radius),
        scale=(1, 1, 1)
    )
    # rename and hide the added circle
    cam_circle = bpy.context.object
    cam_circle.name = 'cam_circle'
    cam_circle.hide_render = True
    cam_circle.hide_viewport = True

    # snap camera to random location on camera circle
    obj = bpy.data.objects['cam_circle']  # get circle
    me = obj.data
    me.calc_loop_triangles()  # calculate triangles on cirlce

    mesh_face = []

    # loop on camera cirlce surface
    for tri in me.loop_triangles:
        for i in range(3):
            vert_index = tri.vertices[i]

            mesh_face.append(me.vertices[vert_index].co)

    # select random location on circle face
    rand_mesh_face = random.choice(mesh_face)

    # snap camera to random location
    camera.location = obj.matrix_world @ rand_mesh_face

    # check occlusion and replace camera
    occlusion_percentage = occlusion_detector(target_rig, camera)

    iter = 0
    while occlusion_percentage >= 1 and iter < 30:
        # select random location on circle face
        rand_mesh_face = random.choice(mesh_face)

        # snap camera to random location
        camera.location = obj.matrix_world @ rand_mesh_face

        occlusion_percentage = occlusion_detector(target_rig, camera)
        iter += 1

    # set the start and end of the render to the target rig's animation start and finish
    keyframes = []
    anim = target_rig.animation_data
    if anim is not None and anim.action is not None:
        for fcu in anim.action.fcurves:
            for keyframe in fcu.keyframe_points:
                x, y = keyframe.co
                if x not in keyframes:
                    keyframes.append((math.ceil(x)))
    bpy.context.scene.frame_start = min(keyframes)
    bpy.context.scene.frame_end = max(keyframes)

    # creating name tag
    # name_tag = 'RandomCamera_' + \
    #     str(itr) + '_' + scene_name + '_' + \
    #     target_rig.name.replace('Armature: ', '')
    # name_tag = name_tag.replace('RC', f'RC_{itr}_')

    # creating new folder for new camera location
    os.chdir(Dataset_dir)
    os.makedirs(str(name_tag), exist_ok=True)
    os.chdir(os.path.join(Dataset_dir, str(name_tag)))
    bpy.context.scene.render.filepath = os.path.join(os.getcwd(), 'test')

    # set the render properties
    render_setting()

    # calling semantic segmentation function
    Setup_Segmentation(os.path.join(os.getcwd(), 'Semantic Segmentation'))

    # calling depth map function
    Depth_Map_Genrator(os.path.join(os.getcwd(), 'Depth Map'))

    # rendering the animation
    bpy.ops.render.render(animation=True, write_still=True)

    # print('Capturing scene information\n****************************************************\n')
    # calling data capture function
    joint_tracker(lighting, workers_name_list, path=os.getcwd())
    logger.info('Random camera view rendered for ' + target_rig.name + ' in ' + scene_name + ' scene.')


def rendering_drone_view(lighting, camera, scene_name, workers_name_list, name_tag):
    # get the max and min animations keyframe
    workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
    min_keys = []
    max_keys = []
    for worker in workers:
        keyframes = []
        anim = worker.animation_data
        if anim is not None and anim.action is not None:
            for fcu in anim.action.fcurves:
                for keyframe in fcu.keyframe_points:
                    x, y = keyframe.co
                    if x not in keyframes:
                        keyframes.append((math.ceil(x)))
        min_keys.append(min(keyframes))
        max_keys.append(max(keyframes))
    # set the strat and end of the render
    bpy.context.scene.frame_start = min(min_keys)
    bpy.context.scene.frame_end = max(max_keys)

    # get the max and min coordinates of the point-cloud
    max_coord_x, max_coord_y, max_coord_z = float(
        'inf') * -1, float('inf') * -1, float('inf') * -1
    min_coord_x, min_coord_y, min_coord_z = float(
        'inf') * 1, float('inf') * 1, float('inf') * 1
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name != 'Horizon' and 'Horizon' not in obj.name:
            min_mesh_x = min(
                [(obj.matrix_world @ v.co).x for v in obj.data.vertices])
            min_mesh_y = min(
                [(obj.matrix_world @ v.co).y for v in obj.data.vertices])
            min_mesh_z = min(
                [(obj.matrix_world @ v.co).z for v in obj.data.vertices])

            max_mesh_x = max(
                [(obj.matrix_world @ v.co).x for v in obj.data.vertices])
            max_mesh_y = max(
                [(obj.matrix_world @ v.co).y for v in obj.data.vertices])
            max_mesh_z = max(
                [(obj.matrix_world @ v.co).z for v in obj.data.vertices])

            if min_mesh_x < min_coord_x:
                min_coord_x = min_mesh_x
            if min_mesh_y < min_coord_y:
                min_coord_y = min_mesh_y
            if min_mesh_z < min_coord_z:
                min_coord_z = min_mesh_z

            if max_mesh_x > max_coord_x:
                max_coord_x = max_mesh_x
            if max_mesh_y > max_coord_y:
                max_coord_y = max_mesh_y
            if max_mesh_z > max_coord_z:
                max_coord_z = max_mesh_z

    # make a list of curve's control points
    coords_list = [
        [max_coord_x, max_coord_y, max_mesh_z + 30],
        [max_coord_x, min_coord_y, max_mesh_z + 30],
        [min_coord_x, min_coord_y, max_mesh_z + 30],
        [min_coord_x, max_coord_y, max_mesh_z + 30],
        [max_coord_x, max_coord_y, max_mesh_z + 30],
        [max_coord_x, min_coord_y, max_mesh_z + 30],
        [min_coord_x, min_coord_y, max_mesh_z + 30]
    ]

    # remove previous curves from scene
    bpy.ops.object.select_all(action='DESELECT')
    try:
        bpy.data.objects['crv'].select_set(True)
        if bpy.data.objects['crv'].select_get():
            bpy.ops.object.delete(use_global=False, confirm=False)
    except:
        pass

    # make a curve around the point-cloud
    crv = bpy.data.curves.new('crv', 'CURVE')
    crv.dimensions = '3D'

    spline = crv.splines.new(type='NURBS')

    # there's already one point by default
    spline.points.add(len(coords_list)-1)

    # assign the point coordinates to the spline points
    for p, new_co in zip(spline.points, coords_list):
        p.co = (new_co + [1.0])  # (add nurbs weight)

    # make a new object with the curve
    obj = bpy.data.objects.new('crv', crv)
    bpy.context.collection.objects.link(obj)

    # remove previous camera constraints
    for c in camera.constraints:
        camera.constraints.remove(c)

    # add follow_path constraint
    camera.constraints.new(type='FOLLOW_PATH')
    follow_path = camera.constraints.get("Follow Path")
    follow_path.target = bpy.data.objects['crv']
    follow_path.use_fixed_location = True

    follow_path.offset_factor = 0
    follow_path.keyframe_insert(
        "offset_factor", frame=bpy.context.scene.frame_start)

    follow_path.offset_factor = 1
    follow_path.keyframe_insert(
        "offset_factor", frame=bpy.context.scene.frame_end)

    bpy.ops.constraint.followpath_path_animate(
        constraint="Follow Path", owner='OBJECT')

    # add "TRACK_TO" constraint to the camera
    camera.constraints.new(type="TRACK_TO")
    camera.constraints['Track To'].target = bpy.data.objects["Floor"]

    # creating name tag
    # name_tag = 'DroneCamera_' + scene_name

    # creating new folder for new camera location
    os.chdir(Dataset_dir)
    os.makedirs(str(name_tag))
    os.chdir(os.path.join(Dataset_dir , str(name_tag)))
    bpy.context.scene.render.filepath = os.path.join(os.getcwd(), 'test')

    # set the render properties
    render_setting()

    # calling semantic segmentation function
    Setup_Segmentation(os.path.join(os.getcwd(), 'Semantic Segmentation'))

    # calling depth map function
    Depth_Map_Genrator(os.path.join(os.getcwd(), 'Depth Map'))

    # rendering the animation
    bpy.ops.render.render(animation=True, write_still=True)

    logger.info('Drone camera view rendered for ' + scene_name + ' scene.')
    # print('Capturing scene information\n****************************************************\n')
    # calling data capture function
    joint_tracker(lighting, workers_name_list, path=os.getcwd())


"""
Main Body
"""

# with open(report_path, 'w') as f:
logger.info(f'Starting the data generation loop for {Number_of_Image_Sequences} times.')
for i in range(Number_of_Image_Sequences):
    scenepath = random.choice(scene_paths)
    scene_name = scenepath.name.replace('.blend', '')

    "Append horizon"
    with bpy.data.libraries.load(horizon_path.as_posix()) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects]

    # link them to scene
    scene = bpy.context.scene
    for i, obj in enumerate(data_to.objects):
        if obj is not None:
            scene.collection.objects.link(obj)
            obj.name = f'Horizon{i+1}'
    logger.info('Horizon is loaded')
    # print('Horizon is loaded')
    
    #TODO
    # why excluding Camera, light and Plane?
    "Append scene"
    with bpy.data.libraries.load(scenepath.as_posix()) as (data_from, data_to):
        data_to.objects = [name for name in data_from.objects if (
            'Camera' and 'Light' and 'Horizon') not in name]

    # link them to scene
    scene = bpy.context.scene
    for obj in data_to.objects:
        if obj is not None: 
            scene.collection.objects.link(obj)

    # Snap scene to the ground
    scene = [obj for obj in bpy.data.objects if 'Horizon' not in obj.name]

    # find the lowest Z value of scene
    scene_lowest_pt = 1 * float('inf')
    for mesh in scene:
        if mesh.type == 'MESH':
            # get the minimum z-value of all vertices after converting to global transform
            mesh_lowest_pt = min(
                [(mesh.matrix_world @ v.co).z for v in mesh.data.vertices])
            if mesh_lowest_pt < scene_lowest_pt:
                scene_lowest_pt = mesh_lowest_pt

    # snap the scene to the ground
    for obj in bpy.data.objects:
        if 'Horizon' not in obj.name:
            obj.location.z -= scene_lowest_pt        
    logger.info('Scene Loaded:\n' + str(scenepath) + '\n')
    # print('Scene Loaded:\n', scenepath, '\n')
    
    
    
    "Generate sky texture"
    # remove any previous sky_texture

    # add new sky texture
    sky_texture = bpy.context.scene.world.node_tree.nodes.new(
        "ShaderNodeTexSky")
    bg = bpy.context.scene.world.node_tree.nodes["Background"]
    bpy.context.scene.world.node_tree.links.new(
        bg.inputs["Color"], sky_texture.outputs["Color"])
    sky_texture.sky_type = 'NISHITA'

    bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].sun_intensity = 0.1

    "Append avatars to scene"

    logger.info('Loading Avatars:\n')
    # print('Loading Avatars:\n')
    for i in range(Number_of_Workers):
        filepath = random.choice(avatar_paths)
        worker_name = filepath.name.replace('.blend', '')
        logger.info(f'Avatar {i}: {worker_name}')
        # print(f'Avatar {i}: {worker_name}')
        with bpy.data.libraries.load(filepath.as_posix()) as (data_from, data_to):
            data_to.objects = [name for name in data_from.objects if (
                'Camera' and 'Light' and 'Horizon') not in name]

        # link them to scene
        scene = bpy.context.scene
        for obj in data_to.objects:
            if obj is not None:
                scene.collection.objects.link(obj)
            if 'Armature' in obj.name:
                obj.name = obj.name + ': ' + worker_name
        
        # set the origin of armature to the lowest point
        rig = bpy.data.objects[obj.name]
        # find the lowest Z value of rig
        rig_lowest_pt = 1*float('inf')

        for obj in bpy.data.objects:
            if (obj.parent == rig and obj.type == 'MESH'):

                # get the minimum z-value of all vertices after converting to global transform
                mesh_lowest_pt = min(
                    [(obj.matrix_world @ v.co).z for v in obj.data.vertices])

                if mesh_lowest_pt < rig_lowest_pt:
                    rig_lowest_pt = mesh_lowest_pt
        rig.location.z += rig.location.z - rig_lowest_pt
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        

    "Localize avatars in scene"
    # create a list of floor planes
    planes = [obj for obj in bpy.data.objects if 'Floor' in obj.name]

    # hide floor planes from render view
    for p in planes:
        p.hide_render = True

    # choose a random floor plane
    plane = random.choice(planes)

    # choose a random location on selected floor
    me = plane.data
    me.calc_loop_triangles()  # calculate triangles on floor
    mesh_face = []

    # loop on floor surface
    for tri in me.loop_triangles:
        for i in range(3):
            vert_index = tri.vertices[i]
            mesh_face.append(me.vertices[vert_index].co)
    logger.info('Avatars are loaded')
    # print('Avatars are loaded') 
    
    # snap rigs to ground
    rigs = [obj for obj in bpy.data.objects if 'Armature' in obj.name]

    "Randomize location of avatars in scene for 'Iterations_Avatar_Location_Randomization' times"
    itr = 0
    for i in range(Iterations_Avatar_Location_Randomization):
        logger.info(f'Randomizing Avatars Location; iteration {i+1}')
        for rig in rigs:
            # select random location on floor face
            rand_mesh_face = random.choice(mesh_face)

            # convert random location to world coordinates
            rand_location = plane.matrix_world @ rand_mesh_face

            # snap the rig to the random location
            rig.location = rand_location


        "Randomize the camera and lights for 'Iterations_Lighting_Randomization' times"
        for i in range(Iterations_Lighting_Randomization):
            logger.info(f'Randomizing Lighting; iteration {i+1}')
            itr += 1

            "Randomize lighting condition"
            lighting = {}
            sun_state = random.weibullvariate(1, 1.1)
            if sun_state > 1: sun_state = 1
            sun_elevation = random.gauss(0.5, 0.5/3)
            # if sun_elevation < 0.05: sun_elevation = 0.05
            # if sun_elevation > 0.95: sun_elevation = 0.95
            if sun_elevation < 0.1: sun_elevation = 0.1
            if sun_elevation > 0.9: sun_elevation = 0.9

            # bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].air_density = sun_state * 10
            bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].air_density = sun_state * 5
            bpy.data.worlds['World'].node_tree.nodes['Sky Texture'].sun_elevation = sun_elevation * 1.57
            lighting['sun_state'] = sun_state * 10
            lighting['sun_elevation'] = sun_elevation * 1.57

            "Add camera to the scene"
            new_camera(focal_len=random.choice([15, 20, 30, 40]))
            camera = bpy.data.objects['Camera']

            # select random rig as target rig
            target_rig = random.choice(rigs)
            workers_name_list = [rig.name.replace('Armature: ', '') for rig in rigs]

            logger.info('Rendering random camera for avatar ' + target_rig.name + ' in ' + scene_name + ' scene.')
           
            name_tag = 'RC_' + \
                str(itr) + '_' + scene_name + '_' + \
                target_rig.name.replace('Armature: ', '')

            try:
                rendering_random_camera(
                    lighting=lighting, itr=itr, 
                    workers_name_list=workers_name_list, 
                    camera=camera, 
                    target_rig=target_rig, 
                    scene_name=scene_name, 
                    name_tag=name_tag)
                logger.info(f'Rendering done for {str(name_tag)}')
                # f.write(str(name_tag) + ':\nDone\n********\n')
            except Exception as e:
                logger.error(f'Error in rendering {str(name_tag)}: {str(e)}')
                

    # render drone view
    if Drone_View:
        "Add camera to the scene"
        new_camera(focal_len=random.choice([15, 20, 30, 40]))
        camera = bpy.data.objects['Camera']

        name_tag = 'RandomCamera_' + \
            str(itr) + '_' + scene_name + '_' + \
            target_rig.name.replace('Armature: ', '')
        
        try:
            rendering_drone_view(
                lighting=lighting, 
                workers_name_list=workers_name_list, 
                camera=camera, 
                scene_name=scene_name,
                name_tag=name_tag)
            logger.info(f'Rendering done for {str(name_tag)}')
            
        except Exception as e:
            logger.error(f'Error in rendering {str(name_tag)}: {str(e)}')
            

    bpy.ops.wm.open_mainfile(filepath=empty_dir.as_posix(), display_file_selector=False)
