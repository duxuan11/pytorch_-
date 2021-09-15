import os
import shutil
#根据是否有标签来copy图片


def dfs_doc(lbael_path,file_path):
    #首先遍历当前目录所有文件夹
    file_list = os.listdir(lbael_path)
    file_list2 = os.listdir(file_path)
    for file in file_list:
        filename = file.split('.')[0]
        for file2 in file_list2:
            if filename == file2.split('.')[0]:
                    shutil.copyfile(file_path+'/'+file2, "JPEGImages2/"+file2)
save_img_path = 'JPEGImages' #用来存储照片的名称，以及出错的名称
save_Annotations='Annotations'
dfs_doc(save_Annotations,save_img_path)