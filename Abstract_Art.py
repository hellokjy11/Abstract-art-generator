import pygame
import cv2
import numpy as np
from matplotlib import pyplot as plt
import PIL
import math
from PIL import Image

photos=['image0.jpeg','image1.jpg','image2.jpg','image3.jpg','image4.jpeg','image5.jpg','image6.jpeg',]
PI=math.pi

photo=photos[6] #change the file number from 0 to 6
img = cv2.imread(photo,0)
img = cv2.medianBlur(img,5) #(Abid K, 2014)
img1=cv2.imread(photo)
img2=Image.open(photo)

ret,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
#the different types of threshold (Mordvintsev, A. and Abid, K., 2014)
titles = ['Original Image', 'Global Thresholding',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]




class my_dictionary(dict):
  
    def __init__(self):
        self = dict()
          
    
    def add(self, key, value):
        self[key] = value
        
for i in range(0,4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()
i=1
dec=1
medianFiltered = cv2.medianBlur(th1,5)



cv2.imshow('Median Filtered Image',medianFiltered)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(medianFiltered, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #(S.Perone, C., 2014)
contour_list = []
areas=my_dictionary()
ara=[]
sorted_data=[]

for contour in contours:
    area = cv2.contourArea(contour)
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) #arcLenth- true: means contour not 'open'
    M = cv2.moments(contour)
    #(anon, opencv)
    if M['m00'] != 0.0:
        x = int(M['m10']/M['m00']) #(gerkey, 2015)
        y = int(M['m01']/M['m00'])
    
    if area > 500 :
        contour_list.append(contour)
        cordinate = x, y 
        print(i,x,y,area,len(approx)) #i, x and y coordinates, area of the contoured section and the orientation
        ara.append(area)
        areas.add(area,(x,y,len(approx)))
        R,G,B=img2.getpixel(cordinate) 
        cv2.drawContours(img1, [approx], 0, (R,G,B), 5)
        cv2.putText(img1, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        i=i+1
j=i-1
print(j)
pygame.init()
ara=sorted(ara, key=float,reverse=True)
print(ara)
m=0
print(areas)
wid, hgt = img2.size
#print(wid, hgt)
cv2.drawContours(img1, contour_list,  -1, (0,0,0), 2)
cv2.imshow('Objects Detected Contour',img1)


img1=cv2.imread(photo)
img=img1
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#(Abid K. 2013) 
# vehicle = cv2.CascadeClassifier(r'vehicle.xml')
# Object = cv2.CascadeClassifier(r'object.xml')
vehicle = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
Object = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

vehicles = vehicle.detectMultiScale(gray_img)
Objects=Object.detectMultiScale(gray_img)
#(Mordvintsev, A. and Abid, K., 2014)
for (x,y,w,h) in vehicles:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
    cv2.putText(img, 'vehicle', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

for (x,y,w,h) in Objects:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)
    cv2.putText(img, 'Object', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.imshow('detection',img)
cv2.waitKey()

for i in range(0,j):
    sorted_data.append(areas[ara[i]])

display_surface = pygame.display.set_mode((wid, hgt))
pygame.display.set_caption('Abstract Art')
cordinate = 250,50
col=img2.getpixel(cordinate)
white = (255, 255, 255)
green = (0, 255, 0)
blue = (0, 0, 128)
black = (0, 0, 0)
red = (255, 0, 0)
display_surface.fill(col)

'''
print("Sorted Data X,Y Values:")
print(sorted_data[0][0],sorted_data[0][1])
print(sorted_data[1][0],sorted_data[1][1])
print(sorted_data[2][0],sorted_data[2][1])
print(sorted_data[3][0],sorted_data[3][1])

print("Sorted Data areas Values:")
print(ara[0])
print(ara[1])
print(ara[2])
print(ara[3])

print("Sorted Data Approx Values:")
print(sorted_data[0][2])
print(sorted_data[1][2])
print(sorted_data[2][2])
print(sorted_data[3][2])
'''
LSTX=[200,150,100,50,25,25,25,25,25,25]
LSTY=[250,200,150,100,50,50,50,50,50,50]
it=0
for i in range(0,j):
    lx=LSTX[i]
    ly=LSTY[i]
    print(i)
    cordinate = sorted_data[i][0],sorted_data[i][1]
    col=img2.getpixel(cordinate)
    if sorted_data[i][1] > 400:
        if sorted_data[i][2] >=1  and sorted_data[i][2] < 13:
            pygame.draw.line(display_surface, col,(sorted_data[i][0]-lx, sorted_data[i][1]+ly), (900, 0), 200)
            #pygame.draw.line(display_surface, col,(0, 0), (200, 200), 100)

    else:
        if sorted_data[i][2] >=1  and sorted_data[i][2] < 13:
            pygame.draw.line(display_surface, col,(sorted_data[i][0]+lx, sorted_data[i][1]+ly), (sorted_data[i][0]-lx, sorted_data[i][1]-ly), 100)
            #pygame.draw.line(display_surface, col,(0, 0), (200, 200), 100)
    if sorted_data[i][2] ==14:
        if it==0:
            pygame.draw.polygon(display_surface, col, points=[(sorted_data[i][0]-sorted_data[i][0],sorted_data[i][1]-sorted_data[i][1]), (sorted_data[i][0],sorted_data[i][1]), (sorted_data[i][0]-sorted_data[i][0],sorted_data[i][1]+ly)])
            it=1
        elif it==1:
            pygame.draw.polygon(display_surface, col, points=[(sorted_data[i][0]+(4*lx),sorted_data[i][1]-(0.5*ly)), (sorted_data[i][0],sorted_data[i][1]+ly), (sorted_data[i][0]+(4*lx),sorted_data[i][1]+(2*ly))])
            

    if (sorted_data[i][2] >14  and sorted_data[i][2] < 15):
        
        pygame.draw.ellipse(display_surface, col,(sorted_data[i][0]-lx, sorted_data[i][1]-ly, sorted_data[i][0]+lx, sorted_data[i][1]+ly))
        print(sorted_data[i][0]-lx, sorted_data[i][1]-ly, sorted_data[i][0]+lx, sorted_data[i][1]+ly)
        
    if sorted_data[i][2] >=15  and sorted_data[i][2] < 25:
        if dec >1:
            pygame.draw.circle(display_surface,col, (sorted_data[i][0]+lx, sorted_data[i][1]+ly), lx, ly)
            
        else:
            pygame.draw.circle(display_surface,col, (sorted_data[i][0]-lx, sorted_data[i][1]), lx, ly)
        dec=dec+1
            
    m=m+1

if m<5:
    cord=(200,350)
    col=img2.getpixel(cord)
    pygame.draw.ellipse(display_surface, col,(200, 200, 150, 50))
else:
    m=0

while True :
    
    pygame.display.flip()
    
    for event in pygame.event.get() :# iteration for the .event.get outcome
  
        # if type of the event obj ==quit
        pygame.display.flip()
        if event.type == pygame.QUIT :
  
            pygame.quit() 
        pygame.display.update() #draw to screen

pygame.display.flip()
cv2.waitKey()













