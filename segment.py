import torch
import math
import cv2
import numpy as np  
# load image

imagepath = r"C:\Users\Artha Tavshikar\Downloads\backnoseg.jpg"
scanpath = r"D:\Head training\internetbackcolor.ply"

img = cv2.imread(imagepath)
img[np.all(img == (153, 53, 255), axis=-1)] = (0,0,0)

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) 
# set lower and upper color limits
lower_val = np.array([51,107,0])
upper_val = np.array([95,255,255])
# Threshold the HSV image 
mask = cv2.inRange(hsv, lower_val, upper_val)
# remove noise
kernel =  np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
# find contours in mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


#show image
res = cv2.bitwise_and(img,img, mask= mask)

# Read image
im = res

# Set up the detector with default parameters.

params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10    # the graylevel of images
params.maxThreshold = 200

params.filterByColor = True
params.blobColor = 255

params.filterByInertia = False
params.filterByConvexity = False
# Filter by Area
params.filterByArea = False
params.minArea = 10000

detector = cv2.SimpleBlobDetector(params)

ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else : 
	detector = cv2.SimpleBlobDetector_create(params)

contmask = np.zeros(im.shape, np.uint8)

# Detect blobs.
keypoints = detector.detect(im)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(contmask, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()   

mean = cv2.mean(im, mask=mask)

rows, cols, colors = im.shape
rows = rows - 1
cols = cols - 1
cvfirst = 0
cvsecond = 0
imgrgb = []
k = 0
topdistance = 0
i = 0
for y in range(cols):
    for x in range(rows):
        if ((im_with_keypoints[x, y]!= [0,0,0]).any()):
            coords = [y, x]
            imgrgb.append(coords)
            if k == 0:
                first = imgrgb[k]
                second = coords
                distance = math.sqrt((second[-2]-first[-2])**2+(second[-1]-first[-1])**2)
            else:
                for i in range(len(imgrgb)):
                    first = imgrgb[i]
                    second = coords
                    distance = math.sqrt((second[-2]-first[-2])**2+(second[-1]-first[-1])**2)
                    if distance>topdistance:
                        topdistance = distance
                        cvfirst = first
                        cvsecond = second
            k = k+1            
print (cvfirst, cvsecond)

import open3d as o3d
import numpy as np
from open3d import *    
from matplotlib import pyplot as plt


cloud = o3d.io.read_point_cloud(scanpath) # read the point , change path to match yours
pcd = cloud
#the commented code can be used to pick a point and get the coordinates for it 
#mesh = cloud.compute_convex_hull()

#mesh.filter_smooth_laplacian()
downcloud = cloud.voxel_down_sample(voxel_size=5)
                               
#custom_draw_geometry_load_option(cloud)
#print (o3d.geometry.PointCloud.get_center(cloud))
vis = o3d.visualization.VisualizerWithEditing()
vis.create_window()
#print(cloud[:,:,1,:])
vis.add_geometry(downcloud)
vis.run()  # user picks points
vis.destroy_window()
#print("")
#vis.get_picked_points()

import re
coordinates = [] #this is an array containing all the coordinates starting from point1
rgbvalues = [] #this is an array containing all the rgb values starting from point1
coordinates = np.asarray(downcloud.points)
string = str(cloud)
string = re.sub('[^0-9]', '', string)
rgbvalues = np.asarray(downcloud.colors)
#print (rgbvalues)
#print (len(coordinates))
list = []
k = 0
topdistance3d = 0

for i in range(len(rgbvalues)):
    color = rgbvalues[i]
    red = color[-3]
    green = color[-2]
    blue = color[-1]
    if ((green-red)*255>75 and (green-blue)*255>75):
        list.append(i)
        if k == 0:
            first = coordinates[list[0]]
            second = coordinates[list[0]]
            distance = math.sqrt((second[-3]-first[-3])**2+(second[-2]-first[-2])**2)            
        else:
            for j in range(len(list)):
                first = coordinates[list[j]]
                second = coordinates[i]
                distance = math.sqrt((second[-3]-first[-3])**2+(second[-2]-first[-2])**2)
                if distance>topdistance3d:
                    topdistance3d = distance
                    pcfirst = first
                    pcsecond = second
        k = k+1
        
#print(topdistance3d, topdistance)


src = img

#percent by which the image is resized
scalefactor = topdistance3d/topdistance
print(topdistance3d, topdistance)
#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scalefactor)
height = int(src.shape[0] * scalefactor)

# dsize
dsize = (width, height)

# resize image
output = cv2.resize(src, dsize)


    
cvy = cvfirst[-1]
cvx = cvfirst[-2]


openy = pcfirst[-2]
openx = pcfirst[-3]

xoffset = openx - cvx
yoffset = openy - cvy

print("Coordinates need to be scaled by: ", scalefactor, "X values need to be moved by: ", xoffset*scalefactor, "Y values need to be moved by: ", yoffset*scalefactor)
    
    
    
#cv2.imshow('resize',output)         
#cv2.waitKey(0)
#cv2.destroyAllWindows()      
    





from glob import glob
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
import torch
from nets.MobileNetV2_unet import MobileNetV2_unet


# load pre-trained model and weights
def load_model():
    model = MobileNetV2_unet(None).to(args.device)
    state_dict = torch.load(r"C:\Users\Artha Tavshikar\Desktop\python\face-seg-master\face-seg-master\scratch\249\models\best.pt", map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    import argparse
    import matplotlib
    matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description='Semantic Segmentation')


    args = parser.parse_args()
    args.device = torch.device("cpu")

    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    print('Model loaded')



    image = cv2.imread(imagepath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    src = image

    #percent by which the image is resized
    scalefactorw = src.shape[1]/224
    scalefactorh = src.shape[0]/224
  
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(args.device)

    # Forward Pass
    logits = model(torch_img)
    mask = np.argmax(logits.data.cpu().numpy(), axis=1)

    # Plot
    #ax.axis('off')
    #ax.imshow(image.squeeze())

    ax1 = plt.subplot(2, 1, 2 * 0 + 2)
    ax1.axis('off')
    ax1.imshow(mask.squeeze())
    plt.savefig('results.png')
    
    image = cv2.imread(r"C:\Users\Artha Tavshikar\Desktop\python\face-seg-master\face-seg-master\results.png")

    

    #percent by which the image is resized
    leftx = 0
    rightx = 0
    bottomy = 0
    rows, cols, colors = image.shape
    for y in range(cols):
        for x in range(rows):
            if (leftx == 0):
                if ((image[x, y] != [255, 255, 255]).any()):
                    leftx = y
                    topy = x             
    cropped_image = image[topy:image.shape[1], leftx:image.shape[0]]
    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)

    
    
    leftx = 0
    rightx = 0
    bottomy = 0
    rows, cols, colors = cropped_image.shape
    for y in range(cols):
        for x in range(rows):
            if (leftx == 0):
                if ((cropped_image[x, y] != [255, 255, 255]).any()):
                    leftx = y
                    topy = x  
                    
    cropped_image = cropped_image[topy:image.shape[1], leftx:image.shape[0]]
    cropped_image = cv2.rotate(cropped_image, cv2.ROTATE_180)
    image = cropped_image
    
    scalefactorw = src.shape[1]/image.shape[1]
    scalefactorh = src.shape[0]/image.shape[0]
    width = int(image.shape[1] * scalefactorw)
    height = int(image.shape[0] * scalefactorh)
    
    # dsize
    dsize = (width, height)
    
    # resize image
    image = cv2.resize(image, dsize)

    cv2.imshow("output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()      


    
    
    
    
