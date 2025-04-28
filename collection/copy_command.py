import os
import subprocess
root_folder = '/home/storage/classwise_collection/laion_4k_cls_5k_samples'

folders = [folder for folder in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, folder))]

start_index = 1304
for folder in folders:
        for file in os.listdir(os.path.join(root_folder, folder)):
                save_name = f'{start_index:05d}.tar'
                cmd=f"mv {os.path.join(root_folder, folder, file)} {os.path.join(root_folder, save_name)}"
                subprocess.call(cmd,shell=True)
                start_index += 1
