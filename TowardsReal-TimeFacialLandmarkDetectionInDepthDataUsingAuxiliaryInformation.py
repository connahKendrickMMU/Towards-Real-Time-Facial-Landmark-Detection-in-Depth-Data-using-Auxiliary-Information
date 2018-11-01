# In[ ]:  
"""

"""  
# The number next to imports show version number, if applicable
import cv2 # 3.30
import os
import numpy # 1.13.3 
import struct 
from mpl_toolkits.mplot3d import Axes3D 
#from sys import byteorder
import matplotlib.pyplot as plt 
import pandas # 0.19.2
#from keras.wrappers.scikit_learn import KerasRegressor   # 2.1.2
import keras # 2.1.2
from array import array
#from numpy import array 
from keras.layers import Input, Dense, Activation 
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Merge
from keras.optimizers import SGD 
from keras.models import model_from_json, Model    
from keras.callbacks import CSVLogger
numpy.random.seed(seed=8)    
from keras.callbacks import ModelCheckpoint
# Keras uses tensorflow backend, version 1.10.0
# In[]: data reading and processing
    
# depth images are store as binary ushorts, thus we use 16 bit struct for quick reading
def ReadDepthImage(Path):
    HoldFullDepthImage = numpy.zeros(424*512)
    with open(Path[0], 'rb') as f: 
        for p in range (424*512):
            HoldFullDepthImage[p] = struct.unpack('H',f.read(2))[0]
    shapedDepthImage = HoldFullDepthImage.reshape((424,512))  
    return shapedDepthImage   

# rgb images are stored as 8 bit RGBA images
def ReadRGBImage(Path):  
    a = array('B')
    a.fromfile(open(Path[0], 'rb'), os.path.getsize(Path[0]) // a.itemsize)  
    a = numpy.array(a) 
    a = a.reshape((1080,1920,4))
    return a/255 

# an example of our csv file for data reading is provided 
def ReadAnExpressionFrame(pathToFile, count):
    df = pandas.read_csv(os.path.expanduser(pathToFile))  
    
    if(df.shape[0] < 1):
        return None,None,None,None,None,None
    DepthPaths = numpy.vstack(df['DepthPath'].values) 
    RGBPaths = numpy.vstack(df['RGBPath'].values)  
    
    DepthImage = ReadDepthImage(DepthPaths[0]).copy()  
    RGBImage = ReadRGBImage(RGBPaths[0]).copy()
    
    #Points = numpy.zeros((df.shape[0],df.shape[0]-7)) 
    RGBBoxs = numpy.zeros((df.shape[0],4))
    DepthBoxs = numpy.zeros((df.shape[0],4))  
    
    UVPoints = numpy.zeros((df.shape[0],64))
    XYZPoints = numpy.zeros((df.shape[0],96)) 
    
    RGBFaces = numpy.zeros((df.shape[0],96,96,3)) 
    DepthFaces = numpy.zeros((df.shape[0],96,96,1)) 
    
    for i in range(df.shape[0]):  
        count = count + 1
        print("\rProcessing Image " + str(i) + " of " + str(df.shape[0]), end=" ")  
        #print(DepthPaths[i]) 
        #print(RGBPaths[i])
        DepthImage = ReadDepthImage(DepthPaths[i]).copy()  
        RGBImage = ReadRGBImage(RGBPaths[i]).copy()
        for p in range(4):
            RGBBoxs[i][p] = numpy.vstack(df['RGBFaceRect'+str(p)].values)[i]
            DepthBoxs[i][p] = numpy.vstack(df['DepthFaceRect'+str(p)].values)[i]    
        counterUV = 0 
        counterXYZ = 0
        for p in range(0,136,2):  
            if p < 16 or (p > 16 and p < 34) or p == 46 or p == 50 or p == 36 or p == 40 or p == 98 or p == 100 or p == 104 or p == 106 or p == 110 or p == 112 or p == 116 or p == 118 or p == 122 or p == 126 or p == 130 or p == 134 or p == 58 or (p > 62 and p < 70): 
                continue 
            UVPoints[i][counterUV] = numpy.vstack(df['RGBU'+str(p)].values)[i] 
            UVPoints[i][counterUV+1] = numpy.vstack(df['RGBV'+str(p)].values)[i]  
            XYZPoints[i][counterXYZ] = numpy.vstack(df['CameraX'+str(p)].values)[i]  
            XYZPoints[i][counterXYZ+1] = numpy.vstack(df['CameraY'+str(p)].values)[i] 
            XYZPoints[i][counterXYZ+2] = numpy.vstack(df['CameraZ'+str(p)].values)[i]
            counterUV += 2  
            counterXYZ += 3    
            # bottom right
            #cv2.circle(RGBImage ,(int(RGBBoxs[0][2]),int(RGBBoxs[0][3])),6,(0,0,0),-1)
            # top left
            #cv2.circle(RGBImage ,(int(RGBBoxs[0][0]),int(RGBBoxs[0][1])),6,(0,0,0),-1)
    #    print(int(UVPoints[i][0]))
    #    print(int(UVPoints[i][1]))
    #    print(int(UVPoints[i][2]))
    #    print(int(UVPoints[i][3]))
        RGBImageCrop = RGBImage[int(RGBBoxs[i][1]):int(RGBBoxs[i][1]+(RGBBoxs[i][3]-RGBBoxs[i][1])),int(RGBBoxs[i][0]):int(RGBBoxs[i][0]+ (RGBBoxs[i][2]-RGBBoxs[i][0]))]  
        RGBImageCrop = cv2.resize(RGBImageCrop,(96,96)) 
        RGBImageCrop = numpy.delete(RGBImageCrop, 3, axis=2)  
        DepthImageCrop = DepthImage[int(DepthBoxs[i][1]):int(DepthBoxs[i][1]+(DepthBoxs[i][3]-DepthBoxs[i][1])),int(DepthBoxs[i][0]):int(DepthBoxs[i][0]+ (DepthBoxs[i][2]-DepthBoxs[i][0]))] 
        DepthImageCrop = cv2.resize(DepthImageCrop,(96,96))
        DepthImageCrop = DepthImageCrop.reshape(96,96,1)
        for q in range(0,UVPoints[i].shape[0],2): 
            #cv2.circle(RGBImage,(int(UVPoints[i][q]),int(UVPoints[i][q+1])),4,(0,0,0),-1) 
            UVPoints[i][q] = int((UVPoints[i][q]-RGBBoxs[i][0])*(96/(RGBBoxs[i][2]-RGBBoxs[i][0]))) 
            UVPoints[i][q+1] = int((UVPoints[i][q+1]-RGBBoxs[i][1])*(96/(RGBBoxs[i][3]-RGBBoxs[i][1])))
            #cv2.circle(RGBImageCrop,(int(UVPoints[i][q]),int(UVPoints[i][q+1])),2,(0,0,0),-1)   
            #cv2.circle(DepthImageCrop,(int(UVPoints[i][q]),int(UVPoints[i][q+1])),2,(0,0,0),-1)  
            #(int((UVPoints[i][q]-RGBBoxs[i][0])*(96/(RGBBoxs[i][3]-RGBBoxs[i][1]))),int((UVPoints[i][q+1]-RGBBoxs[i][1])*(96/(RGBBoxs[i][2]-RGBBoxs[i][0]))) 
        DepthFaces[i] = DepthImageCrop 
        RGBFaces[i] = RGBImageCrop 
    #cv2.imshow("DepthFace", DepthImageCrop[0])
    #cv2.imshow("Face",RGBImageCrop[0])
    #cv2.imshow("RGB", RGBImage)
    #cv2.imshow("Depth", DepthImage) 
    #cv2.waitKey(1)
    return RGBBoxs, DepthBoxs, UVPoints, XYZPoints, RGBFaces, DepthFaces, count 

# returns a full scale RGB image as well
def ReadAnExpressionFrameWithFullRGB(pathToFile):
    df = pandas.read_csv(os.path.expanduser(pathToFile))  
    
    if(df.shape[0] < 1):
        return None,None,None,None,None,None
    DepthPaths = numpy.vstack(df['DepthPath'].values) 
    RGBPaths = numpy.vstack(df['RGBPath'].values)  
    
    DepthImage = ReadDepthImage(DepthPaths[0]).copy()  
    RGBImage = ReadRGBImage(RGBPaths[0]).copy()
    
    RGBBoxs = numpy.zeros((df.shape[0],4))
    DepthBoxs = numpy.zeros((df.shape[0],4))  
    
    UVPoints = numpy.zeros((df.shape[0],64))
    XYZPoints = numpy.zeros((df.shape[0],96)) 
    
    RGBFaces = numpy.zeros((df.shape[0],96,96,3)) 
    DepthFaces = numpy.zeros((df.shape[0],96,96,1)) 
    
    RGBImages = numpy.zeros((df.shape[0],1080,1920,4))
    for i in range(df.shape[0]): #df.shape[0]   
        print("\rProcessing Image " + str(i) + " of " + str(df.shape[0]), end=" ")  
        DepthImage = ReadDepthImage(DepthPaths[i]).copy()  
        RGBImage = ReadRGBImage(RGBPaths[i]).copy()
        RGBImages[i] = RGBImage.copy()
        for p in range(4):
            RGBBoxs[i][p] = numpy.vstack(df['RGBFaceRect'+str(p)].values)[i]
            DepthBoxs[i][p] = numpy.vstack(df['DepthFaceRect'+str(p)].values)[i]    
        counterUV = 0 
        counterXYZ = 0
        for p in range(0,136,2):  
            if p < 16 or (p > 16 and p < 34) or p == 46 or p == 50 or p == 36 or p == 40 or p == 98 or p == 100 or p == 104 or p == 106 or p == 110 or p == 112 or p == 116 or p == 118 or p == 122 or p == 126 or p == 130 or p == 134 or p == 58 or (p > 62 and p < 70): 
                continue 
            UVPoints[i][counterUV] = numpy.vstack(df['RGBU'+str(p)].values)[i] 
            UVPoints[i][counterUV+1] = numpy.vstack(df['RGBV'+str(p)].values)[i]  
            XYZPoints[i][counterXYZ] = numpy.vstack(df['CameraX'+str(p)].values)[i]  
            XYZPoints[i][counterXYZ+1] = numpy.vstack(df['CameraY'+str(p)].values)[i] 
            XYZPoints[i][counterXYZ+2] = numpy.vstack(df['CameraZ'+str(p)].values)[i]
            counterUV += 2  
            counterXYZ += 3    
        RGBImageCrop = RGBImage[int(RGBBoxs[i][1]):int(RGBBoxs[i][1]+(RGBBoxs[i][3]-RGBBoxs[i][1])),int(RGBBoxs[i][0]):int(RGBBoxs[i][0]+ (RGBBoxs[i][2]-RGBBoxs[i][0]))]  
        RGBImageCrop = cv2.resize(RGBImageCrop,(96,96)) 
        RGBImageCrop = numpy.delete(RGBImageCrop, 3, axis=2)  
        DepthImageCrop = DepthImage[int(DepthBoxs[i][1]):int(DepthBoxs[i][1]+(DepthBoxs[i][3]-DepthBoxs[i][1])),int(DepthBoxs[i][0]):int(DepthBoxs[i][0]+ (DepthBoxs[i][2]-DepthBoxs[i][0]))] 
        DepthImageCrop = cv2.resize(DepthImageCrop,(96,96))
        DepthImageCrop = DepthImageCrop.reshape(96,96,1)
        for q in range(0,UVPoints[i].shape[0],2): 
            UVPoints[i][q] = int((UVPoints[i][q]-RGBBoxs[i][0])*(96/(RGBBoxs[i][2]-RGBBoxs[i][0]))) 
            UVPoints[i][q+1] = int((UVPoints[i][q+1]-RGBBoxs[i][1])*(96/(RGBBoxs[i][3]-RGBBoxs[i][1])))
        DepthFaces[i] = DepthImageCrop 
        RGBFaces[i] = RGBImageCrop 
    return RGBBoxs, DepthBoxs, UVPoints, XYZPoints, RGBFaces, DepthFaces, RGBImages

# In[]: Read in files
print("Loading Training Data")
X = numpy.zeros(0)
X2 = numpy.zeros(0)
y2D = numpy.zeros(0)
y3D = numpy.zeros(0)  
# end with / 
path = r'C:\Users\12102083\Desktop\ForGitHub/'
for file in os.listdir(path):  
    if file.endswith(".csv"): 
        print("\rProcessing Training file : " + path + file, end=" ") 
        UVBoxLabels, DepthBoxLabels, UVLandmarks, XYZLandmarks, RGBImages, DepthImages, count = ReadAnExpressionFrame(path+file,0) 
        if type(RGBImages) is not None:
            if X.any():
                X = numpy.append(X,RGBImages,axis=0) 
                X2 = numpy.append(X2,DepthImages,axis=0)
                y2D = numpy.append(y2D,UVLandmarks,axis=0)
                y3D = numpy.append(y3D,XYZLandmarks,axis=0)  
            else: 
                X = RGBImages
                X2 = DepthImages
                y2D = UVLandmarks
                y3D = XYZLandmarks  
# In[ ]: Declare models
def RGBModelUV():   
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='main_output')(model_dense2) 
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model    

def GreyModelUV():  
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='main_output')(model_dense2) 
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model  

def RGBModelXYZ(): 
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(96,name='main_output')(model_dense2) 
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model   

def GreyModelXYZ(): 
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(96,name='main_output')(model_dense2) 
    model = Model(inputs=model_input, outputs=model_output)
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model   

def RGBModelUVXYZ(): 
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='uv_output')(model_dense2)  
    
    model_dense21 = Dense(1000, activation='relu')(model_flat) 
    model_dense22 = Dense(500,activation='relu')(model_dense21) 
    model_output2 = Dense(96,name='xyz_output')(model_dense22)
    
    model = Model(inputs=model_input, outputs=[model_output,model_output2])
    
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model      

def GreyModelUVXYZ(): 
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2)
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(model_pool1)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='uv_output')(model_dense2) 
    
    model_dense21 = Dense(1000, activation='relu')(model_flat) 
    model_dense22 = Dense(500,activation='relu')(model_dense21) 
    model_output2 = Dense(96,name='xyz_output')(model_dense22)
    
    model = Model(inputs=model_input, outputs=[model_output,model_output2])
    model.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    model.summary()  
    return model  

def RGBMergeModelUV(): 
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='main_output')(model_dense2) 
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=model_output)
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()   
    return modelMerge   

def GreyMergeModelUV():
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='main_output')(model_dense2) 
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=model_output)
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()   
    return modelMerge 

def RGBMergeModelXYZ():
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(96,name='main_output')(model_dense2) 
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=model_output)
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()   
    return modelMerge   

def GreyMergeModelXYZ():
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(96,name='main_output')(model_dense2) 
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=model_output)
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()   
    return modelMerge  

def RGBMergeModelUVXYZ():
    model_input = Input(shape=(96,96,3), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='uv_output')(model_dense2) 
    
    model_dense21 = Dense(1000, activation='relu')(model_flat) 
    model_dense22 = Dense(500,activation='relu')(model_dense21) 
    model_output2 = Dense(96,name='xyz_output')(model_dense22)
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=[model_output,model_output2])
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()   
    return modelMerge    

def GreyMergeModelUVXYZ():
    model_input = Input(shape=(96,96,1), dtype='float', name='main_input')
    model_input2 = Input(shape=(96,96,1), dtype='float', name='depth_input')
    
    model_conv1 = Convolution2D(32,(3,3),activation='relu')(model_input)  
    model_conv2 = Convolution2D(32,(2,2),activation='relu')(model_conv1)  
    model_pool1 = MaxPooling2D((2,2))(model_conv2) 
    
    model_conv2_1 = Convolution2D(32,(3,3),activation='relu')(model_input2)  
    model_conv2_2 = Convolution2D(32,(2,2),activation='relu')(model_conv2_1)  
    model_pool2_1 = MaxPooling2D((2,2))(model_conv2_2) 
    
    sumed = keras.layers.Add()([model_pool1, model_pool2_1])
    
    model_conv3 = Convolution2D(32,(3,3),activation='relu')(sumed)   
    model_conv4 = Convolution2D(32,(2,2),activation='relu')(model_conv3)  
    model_conv5 = Convolution2D(32,(2,2),activation='relu',name='conv_final')(model_conv4) 
    model_pool2 = MaxPooling2D((2,2))(model_conv5)
    model_flat = Flatten()(model_pool2)
    
    model_dense1 = Dense(1000, activation='relu')(model_flat) 
    model_dense2 = Dense(500, activation='relu')(model_dense1)  
    model_output = Dense(64,name='uv_output')(model_dense2) 
    
    model_dense21 = Dense(1000, activation='relu')(model_flat) 
    model_dense22 = Dense(500,activation='relu')(model_dense21) 
    model_output2 = Dense(96,name='xyz_output')(model_dense22)
    
    modelMerge = Model(inputs=[model_input,model_input2], outputs=[model_output,model_output2])
    
    modelMerge.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae'])    
    modelMerge.summary()    
    return modelMerge  


# In[ ]: Train the networks
model = None
####################################### all models
print("Fitting Model")  
print("RGBDALL ")
model = RGBMergeModelUVXYZ() 
csv_logger = CSVLogger(r'Logs\RGBD_UVXYZ.csv', append=True, separator=',')
hist = model.fit([X, X2], [y2D,y3D], nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
# For val do this : hist = model.fit([X, X2], [y2D,y3D], nb_epoch=100, validation_data=([ValX,ValX2],[Valy2D,Valy3D]),callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\RGBD_UVXYZ.json','w').write(json_string)   
model.save('RGBD_UVXYZ.h5')
print("data saved")  

model = None 

print("Fitting Model")  
print("RGBDALL ")
model = RGBMergeModelUV()
csv_logger = CSVLogger(r'Logs\RGBD_UV.csv', append=True, separator=',')
hist = model.fit([X, X2], y2D, nb_epoch=100,callbacks=[csv_logger], batch_size=100, verbose = 1)   
json_string = model.to_json()  
print("saving Data")
open(r'Logs\RGBD_UV.json','w').write(json_string)   
model.save('RGBD_UV.h5')
print("data saved")  

model = None 

print("Fitting Model")  
print("RGBDALL ")
model = RGBMergeModelXYZ()
csv_logger = CSVLogger(r'Logs\RGBD_XYZ.csv', append=True, separator=',')
hist = model.fit([X, X2], y3D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\RGBD_XYZ.json','w').write(json_string)   
model.save('RGBD_XYZ.h5')
print("data saved")  

model = None 

####################################### RGB models
print("Fitting Model")  
model = RGBModelUVXYZ()
csv_logger = CSVLogger(r'Logs\RGB_UVXYZ.csv', append=True, separator=',')
hist = model.fit(X, [y2D,y3D], nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\RGB_UVXYZ.json','w').write(json_string)   
model.save('RGB_UVXYZ.h5')
print("data saved")  

model = None 

print("Fitting Model")  
model = RGBModelUV()
csv_logger = CSVLogger(r'Logs\RGB_UV.csv', append=True, separator=',')
hist = model.fit(X, y2D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\RGB_UV.json','w').write(json_string)   
model.save('RGB_UV.h5')
print("data saved")
model = None 

print("Fitting Model")  
print("RGBDALL ")
model = RGBModelXYZ()
csv_logger = CSVLogger(r'Logs\RGB_XYZ.csv', append=True, separator=',')
hist = model.fit(X, y3D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\RGB_XYZ.json','w').write(json_string)   
model.save('RGB_XYZ.h5')
print("data saved")  

model = None

####################################### Depth models
print("Fitting Model")  
model = GreyModelUVXYZ()
csv_logger = CSVLogger(r'Logs\D_UVXYZ.csv', append=True, separator=',')
hist = model.fit(X2, [y2D,y3D], nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\D_UVXYZ.json','w').write(json_string)   
model.save('D_UVXYZ.h5')
print("data saved")  

model = None 

print("Fitting Model")  
model = GreyModelUV()
csv_logger = CSVLogger(r'Logs\D_UV.csv', append=True, separator=',')
hist = model.fit(X2, y2D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json()  
print("saving Data")
open(r'Logs\D_UV.json','w').write(json_string)   
model.save('D_UV.h5')
print("data saved")  

model = None 

print("Fitting Model")  
print("RGBDALL ")
model = GreyModelXYZ()
csv_logger = CSVLogger(r'Logs\D_XYZ.csv', append=True, separator=',')
hist = model.fit(X2, y3D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\D_XYZ.json','w').write(json_string)   
model.save('D_XYZ.h5')
print("data saved")  

model = None 

####################################### convert RGB to grey 

tempTrain = numpy.zeros((X.shape[0],96,96,1))
for i in range(X.shape[0]):  
    q = X[i]
    q=q.astype(numpy.float32)
    q = cv2.cvtColor(q,cv2.COLOR_BGR2GRAY) 
    tempTrain[i] = q.reshape((96,96,1))  

X = tempTrain 

####################################### Grey models
print("Fitting Model")  
model = GreyModelUVXYZ()
csv_logger = CSVLogger(r'Logs\G_UVXYZ.csv', append=True, separator=',')
hist = model.fit(X, [y2D,y3D], nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\G_UVXYZ.json','w').write(json_string)   
model.save('G_UVXYZ.h5')
print("data saved")  

model = None 

print("Fitting Model")  
model = GreyModelUV()
csv_logger = CSVLogger(r'Logs\G_UV.csv', append=True, separator=',')
hist = model.fit(X, y2D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\G_UV.json','w').write(json_string)   
model.save('G_UV.h5')
print("data saved")  

model = None 

print("Fitting Model")  
print("RGBDALL ")
model = GreyModelXYZ()
csv_logger = CSVLogger(r'Logs\G_XYZ.csv', append=True, separator=',')
hist = model.fit(X, y3D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\G_XYZ.json','w').write(json_string)   
model.save('G_XYZ.h5')
print("data saved")  

model = None 

####################################### Grey merge models
print("Fitting Model")  
model = GreyMergeModelUVXYZ()
csv_logger = CSVLogger(r'Logs\GD_UVXYZ.csv', append=True, separator=',')
hist = model.fit([X,X2], [y2D,y3D], nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\GD_UVXYZ.json','w').write(json_string)   
model.save('GD_UVXYZ.h5')
print("data saved")  

model = None 

print("Fitting Model")  
model = GreyMergeModelUV()
csv_logger = CSVLogger(r'Logs\GD_UV.csv', append=True, separator=',')
hist = model.fit([X,X2], y2D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\GD_UV.json','w').write(json_string)   
model.save('GD_UV.h5')
print("data saved")  

model = None 

print("Fitting Model")  
print("RGBDALL ")
model = GreyMergeModelXYZ()
csv_logger = CSVLogger(r'Logs\GD_XYZ.csv', append=True, separator=',')
hist = model.fit([X,X2], y3D, nb_epoch=100, callbacks=[csv_logger], batch_size=100, verbose = 1)  
json_string = model.to_json() 
print("saving Data")
open(r'Logs\GD_XYZ.json','w').write(json_string)   
model.save('GD_XYZ.h5')
print("data saved")  

model = None