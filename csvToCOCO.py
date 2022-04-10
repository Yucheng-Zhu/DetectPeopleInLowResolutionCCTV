import pandas as pd
import os
import cv2


# -5- Define functions
# -- 
def read_makesense_ai_output(
    folder,
    makesenseai_files
):
    makesenseai_dfs_list = []
#     folder = 'img/train/'
    for makesenseai_file in makesenseai_files:
        makesenseai_path = os.path.join(folder, makesenseai_file)
        makesenseai_df = pd.read_csv(
            #folder+'labels.csv',
            makesenseai_path,
            header=None,
            names=[
                'class', 
                'x','y','w','h', 
                'file_name', 
                'image_width','image_height'
            ]
        )
        makesenseai_dfs_list.append(makesenseai_df)
    makesenseai_df = pd.concat(
        makesenseai_dfs_list,
        axis=0,
        join='outer'
    ).reset_index()
    return makesenseai_df

# -- 
def get_images_names_range(makesenseai_df, start=0):
    images_count = makesenseai_df['file_name'].unique().shape[0]
    # files = range(args.start, args.start-1+length) # -e.g.- ['0.png', '1.png', ...]
    images_names_range = range(start, start+images_count) # -e.g.- ['0.png', '1.png', ...]
    return images_names_range

from detectron2.structures import BoxMode
def makesenseai_to_COCO(makesenseai_df, folder, start=0):
    images_names_range = get_images_names_range(
        makesenseai_df, start=start
    )
    COCO = []
    i = start-1
    group = makesenseai_df.groupby('file_name')
    for current_file_name, current_file_content in group:
        i += 1
#     for i in images_names_range:
#         current_file_name = str(i)+'.png'
        current_image = {}
        current_image['file_name'] = \
            os.path.join(folder, current_file_name)
        current_image['image_id'] = int(i)
        img = cv2.imread(
            current_image['file_name']
        )
        height, width, channels = img.shape
        current_image['height'] = int(height)
        current_image['width'] = int(width)
        
        
        current_image_rows = makesenseai_df[
            makesenseai_df['file_name'] \
                == current_file_name
        ]
#         if current_image_rows.shape[0] > 0:
        current_image['annotations'] = []
        
        for r in range(current_image_rows.shape[0]):
            label = {}
            row = current_image_rows.iloc[r]
            label['bbox'] = int(row['x']), int(row['y']), \
                            int(row['w']), int(row['h'])
            label['bbox_mode'] = BoxMode.XYWH_ABS
#             label['segmentation'] = [[916.5,515.5,588.5,173.5,]]
            label['category_id'] = 0
            current_image['annotations'].append(label)
        COCO.append(current_image)
    return COCO

# -- 
import json
def save_COCO_file(path_to_save, COCO):
    with open(path_to_save, "w") as file_handle:
        json.dump(
            COCO, file_handle, 
            indent=4, sort_keys=False
        )

# -5- Define variables
train = {
    'folder': 'data/img/train/',
    'makesenseai_files': ['a.csv', 'b.csv']
}
# valid = {
#     'folder': 'data/img/test/',
#     'makesenseai_files': ['test.csv']
# }
start = 0
# -5- call functions for train, valid, test
# for cv in (train, valid):
for cv in (train,):
    
    folder = cv['folder']
    makesenseai_files = cv['makesenseai_files']
#     makesenseai_path = os.path.join(folder, makesenseai_files)
    
    makesenseai_df = read_makesense_ai_output(folder, makesenseai_files)
    
    COCO_train = makesenseai_to_COCO(
            makesenseai_df, folder, start=start
        )
    
    save_COCO_file(
        # folder+'train.json', 
        os.path.join(folder, 'train.json'), 
        COCO_train
    )
    