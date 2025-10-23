import bpy
import os
import csv
from pathlib import Path


print("Running animation retargetting function. \n Please specify the required paths.")
avatar_path = Path(input("Specify the path to the avatars folder."))
animation_path = Path(input("Specify the path to the animations folder."))
root = Path(os.getcwd())
ouput = Path(input("Specify a folder for outputs of the function."))

for avatar_file in os.listdir(avatar_path): 
    if '.blend' in avatar_file:
        for file in os.listdir(animation_path):
            # Save avatar avatar_file name for saving
            avatar_name = avatar_file.replace('.blend','')
            
            
            # Reading avatar avatar_file
            filepath = avatar_path / avatar_file
            
            with bpy.data.libraries.load(filepath.as_posix()) as (data_from, data_to):
                data_to.objects = [name for name in data_from.objects]
    
            # link them to scene
            scene = bpy.context.scene
            for obj in data_to.objects:
                if obj is not None:
                    scene.collection.objects.link(obj)
            
            # Create report file
            report_name = 'Report_' + '_'.join(avatar_name.replace('.blend','_').split('_')[:-1])
            report_path = root / f'{report_name}.csv'
            
            with open(report_path.as_posix(), 'w') as f:
                # create the csv writer
                writer = csv.writer(f)
    
                # Create list of added avatars
                workers = [worker for worker in bpy.data.objects if 'Armature' in worker.name]
                
                # Detect root bone of avatar
                for worker in workers:
                    avatar_root_bones = []
                    for bone in worker.pose.bones:
                        if bone.parent is None:
                            avatar_root_bones.append(bone.name)
                    
                    for bone in avatar_root_bones:
                        if 'leg' not in bone.lower():
                            target_avatar_root_bones = bone
                
                # Write worker name into CSV
                writer.writerow(avatar_name)
                        
                # Save avatar file name for saving
                animation_name = file.replace('.fbx','')
                
                # Create set of objects in the scene
                old_objs = set(bpy.data.objects)
                # old_anim = set(bpy.data.actions)
                
                # Load animation file
                filepath = animation_path / file
                bpy.ops.import_scene.fbx(filepath=filepath.as_posix(), automatic_bone_orientation=True)
                
                # Create a set of newly added objects to scene
                imported_objs = list(set(bpy.data.objects) - old_objs)
                # imported_anim = list(set(bpy.data.actions) - old_anim)[0]
                
                
                # Detect root bone of animation
                for worker in imported_objs:
                    animation_root_bones = []
                    for bone in worker.pose.bones:
                        if bone.parent is None:
                            animation_root_bones.append(bone.name)
                    
                    for bone in animation_root_bones:
                        if 'leg' not in bone.lower():
                            target_animation_root_bones = bone
                
                # Write animation file name into CSV
                writer.writerow(animation_name)
                
                # Create predifined bone list
                bpy.context.scene.rsl_retargeting_armature_source = imported_objs[0]
                bpy.context.scene.rsl_retargeting_armature_target = workers[0]
                bpy.ops.rsl.build_bone_list()
                
                
                # Configure the bone list
                seen = {}
                iter = 0
                for item in bpy.data.scenes['Scene'].rsl_retargeting_bone_list:
                    
                    # Set the root bones
                    if item.bone_name_source == target_animation_root_bones:
                        bpy.data.scenes['Scene'].rsl_retargeting_bone_list[iter].bone_name_target = target_avatar_root_bones
                    
                    # Remove duplicate target bone entries
                    count = seen.get(item.bone_name_target)
                    if not count:
                        count = 0
                    seen[item.bone_name_target] = count + 1
                    for key, value in seen.items():
                        if value > 1:
                            bpy.data.scenes['Scene'].rsl_retargeting_bone_list[iter].bone_name_target = ""
                            seen[key] = value - 1
                    
                    # Write data to CSV
                    row = [item.bone_name_source, item.bone_name_target, item.bone_name_key]
                    writer.writerow(row)
                    iter += 1
                
                
                # Retarget the animation
                bpy.context.scene.frame_current = 0
                bpy.context.scene.rsl_retargeting_auto_scaling = False
                bpy.ops.rsl.retarget_animation()
                
                   
                # Remove imported FBX file (animation)
                bpy.ops.object.select_all(action='DESELECT')
                for item in imported_objs:
                    item.select_set(True)
                bpy.ops.object.delete(use_global=True)
                
                # # Remove imported FBX file (animation)
                # bpy.data.actions.remove(imported_anim)
                
                
                # Save the animated avatar
                file_name = animation_name.replace('.fbx','_') + '_' + avatar_name
                filepath = os.path.join(ouput , file_name) + '.blend'
                bpy.ops.wm.save_as_mainfile(filepath=filepath)            
                writer.writerow('*********************')
            
                #clean up the scene (prepare of new scene set-up)
                for block in bpy.data.meshes:
                    bpy.data.meshes.remove(block)
                for block in bpy.data.materials:
                    bpy.data.materials.remove(block)
                for block in bpy.data.textures:
                    bpy.data.textures.remove(block)
                for block in bpy.data.images:
                    bpy.data.images.remove(block)
                for block in bpy.data.brushes:
                    bpy.data.brushes.remove(block)
                for block in bpy.data.objects:
                    bpy.data.objects.remove(block)
                for block in bpy.data.cameras:
                    bpy.data.cameras.remove(block)
                for block in bpy.data.armatures:
                    bpy.data.armatures.remove(block)
                for block in bpy.data.actions:
                    bpy.data.actions.remove(block)
            
                # Clean the Scene
                bpy.ops.wm.read_homefile(use_empty=True)
        
