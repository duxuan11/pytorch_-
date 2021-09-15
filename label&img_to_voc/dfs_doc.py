import os
import cv2
from xml.dom.minidom import Document

cur_path = os.getcwd() #该程序当前所在路径

src_img_path = 'images' #自己修改当前选定的文件夹
src_label_path = 'labels'

save_img_path = 'JPEGImages\\'
save_img_label = 'JPEGImagesLabels\\' #用来存储照片的名称，以及出错的名称
save_Annotations='Annotations\\'

#图片格式与尺寸转换
def tif_to_jpg(file_path,filename):
        img = cv2.imread(file_path)
        img = cv2.resize(img,(512,385)) #尺寸自己改，可以自己编写函数接口
        cv2.imwrite(save_img_path+filename+".jpg",img)

#txt标签转voc格式
def txt_to_voc(cur_path,filename):
        #如果能打开filename对应的标签文件
        error_img = False
        class_ind = ('0')
        try:
            f = open(save_img_path+filename+'.jpg')
        except IOError:
        #报错，图片对应的标签没有，打印下来
            with open(save_img_label+"error.txt",'a') as f:      
                   f.writelines(save_img_path+filename+'.jpg')
                   f.write("\n")
            error_img = True
            
        if error_img == False:
            img_size = cv2.imread(save_img_path+filename+'.jpg').shape
            f = open(cur_path)
            solit_lines = f.readlines()
            generate_xml(filename,solit_lines,img_size,class_ind)
        error_img = False
    #读取我们之前保存的图片
    

#可遍历所有文件
def dfs_doc(src_path):
    #首先遍历当前目录所有文件夹
    file_list = os.listdir(src_path)
    #循环判断每个元素是文件还是文件夹，是文件的话，是文件夹的话继续遍历
    for file in file_list:
        # 利用os.path.join()方法取得路径全名，并存入cur_path变量，否则每次只能遍历一层目录
        cur_path = os.path.join(src_path,file)
        #判断是否是文件夹
        if os.path.isdir(cur_path):
            dfs_doc(cur_path)
        else :
            #如果不是文件夹 且是能打开的tif图片
            if file.endswith('.tif') and os.path.getsize(cur_path) > 0:
                filename = file.split('.')[0]
                #创建一个存放图片名的文件，用来对应标签文件
                with open(save_img_label+"labels.txt",'a') as f:      
                   f.writelines(filename)
                   f.write("\n")
                tif_to_jpg(cur_path,filename)
            if file.endswith('.txt') and file != 'classes.txt':
                filename = file.split('.')[0]
                txt_to_voc(cur_path,filename)
               

#生成xml文件
def generate_xml(name,split_lines,img_size,class_ind):
    doc = Document() 
    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)
    title = doc.createElement('folder')
    title_text = doc.createTextNode('organoid')
    title.appendChild(title_text)
    annotation.appendChild(title)
    img_name=name+'.jpg'
    title = doc.createElement('filename')
    title_text = doc.createTextNode(img_name)
    title.appendChild(title_text)
    annotation.appendChild(title)
    source = doc.createElement('source')
    annotation.appendChild(source)
    title = doc.createElement('database')
    title_text = doc.createTextNode('The organoid Database')
    title.appendChild(title_text)
    source.appendChild(title)
    title = doc.createElement('annotation')
    title_text = doc.createTextNode('organoid')
    title.appendChild(title_text)
    source.appendChild(title)
    size = doc.createElement('size')
    annotation.appendChild(size)
    title = doc.createElement('width')
    title_text = doc.createTextNode(str(img_size[1]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('height')
    title_text = doc.createTextNode(str(img_size[0]))
    title.appendChild(title_text)
    size.appendChild(title)
    title = doc.createElement('depth')
    title_text = doc.createTextNode(str(img_size[2]))
    title.appendChild(title_text)
    size.appendChild(title)
    for split_line in split_lines:
        line=split_line.strip().split()
        if line[0] in class_ind:
            object = doc.createElement('object')
            annotation.appendChild(object)
            title = doc.createElement('name')
            #title_text = doc.createTextNode(line[0])
            '''
            自己加的内容，可以删除
            '''
            if line[0] == '0':
                title_text = 'organoid'
                title_text = doc.createTextNode(title_text)
            title.appendChild(title_text)
            object.appendChild(title)
            bndbox = doc.createElement('bndbox')
            object.appendChild(bndbox)
            title = doc.createElement('xmin')
            title_text = doc.createTextNode(str(int(float(line[1])*float(img_size[1]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymin')
            title_text = doc.createTextNode(str(int(float(line[2])*float(img_size[0]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('xmax')
            title_text = doc.createTextNode(str(int(float(line[3])*float(img_size[1]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
            title = doc.createElement('ymax')
            title_text = doc.createTextNode(str(int(float(line[4])*float(img_size[0]))))
            title.appendChild(title_text)
            bndbox.appendChild(title)
    #写进文件里
    f = open(save_Annotations+name+'.xml','w')
    f.write(doc.toprettyxml(indent = ''))
    f.close()





dfs_doc(cur_path+"\\"+src_img_path)
dfs_doc(cur_path+"\\"+src_label_path)
