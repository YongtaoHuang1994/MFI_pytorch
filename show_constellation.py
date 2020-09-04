# plot a single pix image
import os
from PIL import Image
import matplotlib.pyplot as plt


#切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 2)
    box_list = []    
    for i in range(0,2):#两重循环，生成9张图片基于原图的位置
        for j in range(0,2):           
            # print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j*item_width,i*item_width,(j+1)*item_width,(i+1)*item_width)
            box_list.append(box)
    image_list = [image.crop(box) for box in box_list]    
    return image_list

#保存
def save_image(image, image_list):
    i = image_list[0] # 0123分别代表四个角落
    i.save('./result/test.jpg')
    plt.figure("Image up left") # 图像窗口名称
    plt.axis('on') # 关掉坐标轴为 off
    plt.imshow(i, cmap='gray')
    plt.show()
        


if __name__ == '__main__':
    img_path = "./data/train/"
    img_type = "QAM4-17dB"
    img_name = img_path+img_type+"/"+img_type+"-1.jpg"
    img = Image.open(img_name)
    #img = img.convert('L') # convert image to black and white
    print(img)
    plt.figure("Image") # 图像窗口名称
    plt.axis('on') # 关掉坐标轴为 off
    #plt.title('image') # 图像题目
    plt.imshow(img, cmap='gray')
    plt.show()

    image_list = cut_image(img)
    save_image(img, image_list)

