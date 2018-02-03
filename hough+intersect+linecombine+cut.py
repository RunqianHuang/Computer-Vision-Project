import cv2 as cv
import numpy as np
import intersecting_point as intersect
from skimage import morphology

def isbound(edge, ix, iy):
    if ix >= 0 and ix < edge.shape[1] and iy >= 0 and iy < edge.shape[0]:
        return True;
    return False;

def islabel(edge, x, y):
    kernelsize = 1;
    ix = int(x);
    iy = int(y);
    for i in range(-1, 2):
        for j in range(-1, 2):
            if isbound(edge, ix + i, iy + j) and edge[iy + j][ix + i] == 255:
                return True;

    return False;

def islabel1(k,edge, x, y):
    ix = int(x);
    iy = int(y);
    for i in range(-k, k+1):
        for j in range(-k, k+1):
            if isbound(edge, ix + i, iy + j) and edge[iy + j][ix + i] == 255:
                return True

    return False;

def findLen(lines, edge):
    threshold = 30;
    newline = [];
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            #     # 绘制一条直线
            a = np.cos(theta);
            b = np.sin(theta);
            x0 = a * rho;
            y0 = b * rho;
            #print(x0, y0);
            #print(b);
            #print(a);

            x = x0;
            y = y0;
            count = 0;
            maxnum = 0;
            num = 0;

            while num < 1000:
                if islabel(edge, x, y) == True:
                    count += 1;
                    maxnum = max(maxnum, count);
                else:
                    count = 0;
                x = x + 2 * (-b);
                y = y + 2 * a;
                num += 1;
            if maxnum > threshold:
                newline.append([[rho, theta]]);
                continue;

            x = x0;
            y = y0;
            count = 0;
            maxnum = 0;
            num = 0;

            while num < 1000:
                if islabel(edge, x, y) == True:
                    count += 1;
                    maxnum = max(maxnum, count);
                else:
                    count = 0;
                x = x - 2 * (-b);
                y = y - 2 * a;
                num += 1;
            if maxnum > threshold:
                newline.append([[rho, theta]]);
    return newline

def cutLen(k, lines, edge, xhi, yhi, threshold1, threshold2):
    point = [];
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            #     # 绘制一条直线
            a = np.cos(theta);
            b = np.sin(theta);
            x0 = a * rho;
            y0 = b * rho;
            if x0<0:
                x0=0
                y0=rho/b
            if y0<0:
                y0=0
                x0=rho/a
            #print(x0, y0);
            #print(b);
            #print(a);

            x = x0;
            y = y0;
            ncount = 0;
            lcount = 0;
            xstart = ystart = xend = yend = 0;
            startflag = endflag = 0

            while x>=0 and x<=xhi and y>=0 and y<=yhi:
                if islabel1(k,edge, x, y) == True:
                    ncount = 0;
                    if startflag == 0:
                        xstart = x
                        ystart = y
                        startflag = 1
                    lcount += 1;
                else:
                    if startflag == 1 and endflag == 0:
                        xend = x
                        yend = y
                    if lcount < threshold2:
                        startflag = 0
                        lcount = 0
                    ncount += 1;
                    if ncount > threshold1 and startflag == 1:
                        endflag = 1
                x = x + 2 * (-b);
                y = y + 2 * a;
                if startflag == 1 and endflag == 0:
                    if lcount>threshold2:
                        xend = x
                        yend = y
                        endflag=1
                if startflag == 1 and endflag == 1:
                    point.append((int(xstart), int(ystart)));
                    point.append((int(xend), int(yend)));
                    ncount = 0
                    lcount = 0
                    startflag = endflag = 0

            x = x0;
            y = y0;
            ncount = 0;
            lcount = 0;
            xstart = ystart = xend = yend = 0;
            startflag = endflag = 0

            while x>=0 and x<=xhi and y>=0 and y<=yhi:
                if islabel1(k,edge, x, y) == True:
                    ncount = 0;
                    if startflag == 0:
                        xstart = x
                        ystart = y
                        startflag = 1
                    lcount += 1;
                else:
                    if startflag == 1 and endflag == 0:
                        xend = x
                        yend = y
                    if lcount < threshold2:
                        startflag = 0
                        lcount = 0
                    ncount += 1;
                    if ncount > threshold1 and startflag == 1:
                        endflag = 1
                x = x - 2 * (-b);
                y = y - 2 * a;
                if startflag == 1 and endflag == 0:
                    if lcount>threshold2:
                        xend = x
                        yend = y
                        endflag=1
                if startflag == 1 and endflag == 1:
                    point.append((int(xstart), int(ystart)));
                    point.append((int(xend), int(yend)));
                    ncount = 0
                    lcount = 0
                    startflag = endflag = 0

    return point

input = "room.jpg"
output = "room-result.jpg"

img=cv.imread(input)
print("Preprocessing...")
result1 = img.copy()
result2 = img.copy()
result3 = img.copy()
result4 = img.copy()
result5 = img.copy()

img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
img = cv.GaussianBlur(img, (3, 3), 0)

thresh1 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)

size=thresh1.shape

for i in range(size[0]):
    for j in range(size[1]):
        if (thresh1[i][j]==255):
            thresh1[i][j]=0
        else:
            thresh1[i][j]=255

kernel1 = np.ones((2,2),np.uint8)
thresh1 = cv.morphologyEx(thresh1, cv.MORPH_OPEN, kernel1)
cv.imwrite("room-adaptive_threshold.jpg", thresh1)

x = cv.Sobel(thresh1, cv.CV_16S, 1, 0)
y = cv.Sobel(thresh1, cv.CV_16S, 0, 1)
absX = cv.convertScaleAbs(x)  # 转回uint8
absY = cv.convertScaleAbs(y)
edges = cv.addWeighted(absX, 0.5, absY, 0.5, 0)
cv.imwrite("room-sobel.jpg", edges)# sobel探测边界

img2=cv.imread("room-sobel.jpg")
#img2=cv.imread("sobelresult.jpg")
ret,thresh = cv.threshold(img2,10,255,cv.THRESH_BINARY)
kernel = np.ones((5,5),np.uint8)
dilation = cv.dilate(thresh,kernel,iterations = 2)
erosion = cv.erode(dilation,kernel,iterations = 2)
#edges = cv.Canny(thresh, 50, 150, apertureSize=3)
#dst=morphology.remove_small_objects(edges,min_size=10000,connectivity=2)
cv.imwrite("room-edge.jpg", erosion)# 形态学处理

img3=cv.imread("room-edge.jpg")
B, G, R = cv.split(img3)
edges = cv.Canny(img3, 50, 150, apertureSize=3)# 边界检测
kernel = np.ones((2,2),np.uint8)
edges = cv.dilate(edges,kernel,iterations = 1)
cv.imwrite("room-canny.jpg", edges)
print("Hough transform...")
lines = cv.HoughLines(edges, 1, np.pi / 36, 75)  # 这里对最后一个参数使用了经验型的值
for i in range(len(lines)):
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

        cv.line(result1,(x1,y1),(x2,y2),(0,0,255),1)
cv.imwrite("room-hough.jpg",result1)
print("Deleting lines...")
newline = findLen(lines,edges)
for i in range(len(newline)):
        for rho,theta in newline[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

        cv.line(result2,(x1,y1),(x2,y2),(0,0,255),1)
cv.imwrite("room-deleteline.jpg",result2)
#print(lines)
# 平行直线合并
print("Combining lines...")
threshold = 35
theta = []
num = 0
number = 0
sum_num = []
rho_new = []
theta_new = []
for line in newline:
    if line[0][1] not in theta:
        theta.append(line[0][1])
        num += 1
for i in range(0,num):
    distance = []
    dnum = 0
    for line in newline:
        if line[0][1]==theta[i]:
            distance.append(line[0][0])
            dnum += 1
    #print(distance)
    for j in range(0,dnum):
        for k in range(j+1,dnum):
            if distance[j]>distance[k]:
                temp = distance[j]
                distance[j]=distance[k]
                distance[k]=temp
    #print(distance)
    tag = []
    tag.append(distance[0])
    tagnum = 1
    for j in range(0,dnum):
        if abs(distance[j]-tag[tagnum-1])>threshold:
            tag.append(distance[j])
            tagnum += 1
    #print(tag)
    for j in range(0,tagnum):
        temp = 0
        sum = 0
        for line in newline:
            if line[0][1]==theta[i]:
                if abs(line[0][0]-tag[j])<=threshold:
                    sum += line[0][0]
                    temp += 1
        avg = sum/temp
        theta_new.append(theta[i])
        rho_new.append(avg)
        sum_num.append(temp)
        number += 1
newlines = []
for i in range(0,number):
    newlines.append([[rho_new[i],theta_new[i]]])
for i in range(len(newlines)):
        for rho,theta in newlines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

        cv.line(result3,(x1,y1),(x2,y2),(0,0,255),1)
cv.imwrite("room-combineline.jpg",result3)
print("Searching intersect points...")
segmented = intersect.segment_by_angle_kmeans(newlines)
intersections = intersect.segmented_intersections(segmented)
for i in range(len(newlines)):
        for rho,theta in newlines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

        cv.line(result4,(x1,y1),(x2,y2),(255,255,255),1)
for center in intersections:
     cv.circle(result4, center, 2, (0, 0, 255), 1)
cv.imwrite("room-intersect.jpg",result4)
print("Cutting lines...")
points = cutLen(6, newlines, edges, size[1], size[0], 0, 20)
numoflen=0
start = []
end = []
for i,p in enumerate(points):
    if i%2==0:
        start.append(p)
    else:
        end.append(p)
        numoflen += 1
for i in range(0,numoflen):
    cv.line(result5, start[i], end[i], (0, 0, 255), 2)
    #cv.line(edges, start[i], end[i], (122), 1)
#for center in intersections:
#     cv.circle(result1, center, 2, (0, 0, 255), 1)
for i in range(len(newlines)):
        for rho,theta in newlines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 2000*(-b))
            y1 = int(y0 + 2000*(a))
            x2 = int(x0 - 2000*(-b))
            y2 = int(y0 - 2000*(a))

        cv.line(edges,(x1,y1),(x2,y2),(122,122,122),1)
#cv.imshow("Result", result1)
cv.imshow('Edge',edges)
#cv.imwrite("finaledge.jpg",edges)
#cv.imwrite(output, result5)
cv.waitKey(0)
cv.destroyAllWindows()