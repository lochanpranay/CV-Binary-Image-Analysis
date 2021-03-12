from matplotlib import pyplot as plt
from numpy import asarray
import os
from os import path
import numpy as np
import math
import cv2 as cv


# Method for thresholding the gray level image.
# Accepts gray level image file object, threshold value, threshold method "threshmode defaults to binary"
# threshmode = "inverse_binary" flips the image foreground and background threshold process.
def threshold(imgFile, thresVal, threshMode='binary'):
    timg = []
    rows = None
    cols = None
    maxIntensity = None
    pixelValues = None

    if imgFile.readline() != "P2\n":
        return []

    for line in imgFile:
        if line.startswith("#"):
            continue
        elif rows == None and cols == None:
            cols, rows = int(line.split()[0]), int(line.split()[1])
        elif maxIntensity == None:
            maxIntensity = int(line)
        else:
            pixelValues = line.split()
            break

    pxcount = 0
    for i in range(rows):
        row = []
        for k in range(cols):
            pix = int(pixelValues[pxcount])
            if threshMode == "binary":
                if pix <= thresVal:
                    row.append(0)
                else:
                    row.append(255)

            if threshMode == "inverse_binary":
                if pix <= thresVal:
                    row.append(255)
                else:
                    row.append(0)
            pxcount = pxcount + 1

        timg.append(row)

    return timg


# Utility method for creating P2 image in the local file system from 2D pixel array.
def createP2Image(pixels, rows, cols, maxIntensity, fileName, comment=None):
    newimg = open(fileName + ".pgm", "w+")
    newimg.write("P2\n")
    if comment != None:
        newimg.write("#" + comment + "\n")
    newimg.write(str(cols) + " " + str(rows) + "\n")
    newimg.write(str(maxIntensity) + "\n")
    for arr in pixels:
        for val in arr:
            newimg.write(str(val) + " ")

    newimg.close()


# A two pass row by row labeling algorithm that labels the forground of the image
# with pixel value starting from "startLabel" on the first pass.
# While lebeling, pushes label conflicts to Union Find DS to generate equivalencies
# On the second pass, replaces all the labels to unique values based on the equivalence array "parent"
# returns a labeled image, startlabel in the image, total components in the image, max intensity label, label set, all label list
def ConnectedComponentLabeling(pixarr):
    startLabel = 50
    currLabel = 50
    equiv = Unionfind(currLabel)

    for i in range(len(pixarr)):
        for k in range(len(pixarr[i])):
            if pixarr[i][k] == 0:
                continue
            nbors = []
            if i > 0 and k > 0:
                nbors = [pixarr[i-1][k], pixarr[i-1][k-1], pixarr[i][k-1]]
                if k + 1 < len(pixarr[i]):
                    nbors.append(pixarr[i-1][k+1])
            elif i < 1 and k > 0:
                nbors = [pixarr[i, k-1]]
            elif i > 0 and k < 1:
                nbors = [pixarr[i-1][k], pixarr[i-1][k+1]]

            nbors = [val for val in nbors if val != 0]

            if len(nbors) == 0:
                pixarr[i][k] = currLabel
                equiv.pushNewLabel(currLabel)
                currLabel += 1
                continue

            nbors.sort()
            pixarr[i][k] = nbors[0]
            if(len(nbors) > 1):
                for lbl in nbors:
                    if lbl == nbors[0]:
                        continue
                    equiv.merge(nbors[0], lbl)

    parent = equiv.getLabelMap()
    totalComps = equiv.getTotalComponents()

    maxPixelVal = 0
    labelset = set()
    for i in range(len(pixarr)):
        for k in range(len(pixarr[i])):
            if pixarr[i][k] == 0:
                continue
            pixVal = parent[pixarr[i][k] - startLabel] + startLabel
            pixarr[i][k] = pixVal
            labelset.add(pixVal)
            if pixVal > maxPixelVal:
                maxPixelVal = pixVal

    return [pixarr, startLabel, totalComps, maxPixelVal, labelset, parent]


# Utility method to convert 2D image array to 3D.
# OpenCV and Matplotlib methods work with only 3D img array.
# However, custom methods in this project works with only 2D arrays. This helps in the conversion.
def convert2dTo3d(arr):
    arr3d = np.zeros((len(arr), len(arr[0]), 3), dtype="uint8")
    for i in range(len(arr)):
        for k in range(len(arr[0])):
            val = arr[i][k]
            arr3d[i][k] = [val, val, val]
    return arr3d


# creates an image with black pixels (mask). Dimensions same as "pixarr"
# Iterates over pixarr searching for the given label "lbl"
# if label found in the image, changes the same position to white(255) on the mask
# returns mask
def extractRegion(lbl, pixarr):
    mask = np.zeros((len(pixarr), len(pixarr[0])), dtype="uint8")
    for i in range(len(pixarr)):
        for k in range(len(pixarr[i])):
            if pixarr[i][k] == lbl:
                mask[i][k] = 255

    return mask


# returns centroid and area
# area = number of foreground pixel in the region
# centroid = sum(rows)/ area, sum(columns) / area
def getCentroidAndArea(pixarr):
    rc = 0
    cc = 0
    area = 0
    rowlen = len(pixarr)
    collen = len(pixarr[0])
    for i in range(rowlen):
        for k in range(collen):
            if pixarr[i][k] == 255:
                rc += i
                area += 1

    for k in range(collen):
        for i in range(rowlen):
            if(pixarr[i][k] == 255):
                cc += k

    centroid = [rc/area, cc/area]
    return [centroid, area]


# Below logic takes the image and iterates over each pixel value
# if a pixel is part of background - ignore and continue
# if a pixel is part of foreground - collect all its neighborhood(N8) values
# if there exists a background pixel in the N8, then record the current pixel as boundry pixel
def getPerimeterPixels(pixarr):
    perimeterCords = []
    rowlen = len(pixarr)
    collen = len(pixarr[0])
    for i in range(rowlen):
        for k in range(collen):
            if pixarr[i][k] == 0:
                continue
            nbors = []
            if i > 0 and k > 0 and i < (rowlen - 1) and k < (collen - 1):
                nbors = [
                    pixarr[i-1][k],
                    pixarr[i-1][k-1],
                    pixarr[i-1][k+1],
                    pixarr[i][k-1],
                    pixarr[i][k+1],
                    pixarr[i+1][k-1],
                    pixarr[i+1][k],
                    pixarr[i+1][k+1]
                ]
            elif i > 0 and k == (collen - 1) and i < (rowlen - 1):
                nbors = [
                    pixarr[i-1][k],
                    pixarr[i+1][k],
                    pixarr[i][k-1],
                    pixarr[i+1][k-1],
                    pixarr[i-1][k-1]
                ]
            elif i == 0 and k == 0:
                nbors = [
                    pixarr[i][k+1],
                    pixarr[i+1][k+1],
                    pixarr[i+1][k]
                ]
            elif i == 0 and k > 0 and k < (collen - 1):
                nbors = [
                    pixarr[i][k+1],
                    pixarr[i+1][k+1],
                    pixarr[i+1][k],
                    pixarr[i+1][k-1],
                    pixarr[i][k-1]
                ]
            elif i == 0 and k == (collen - 1):
                nbors = [
                    pixarr[i][k-1],
                    pixarr[i+1][k-1],
                    pixarr[i+1][k]
                ]
            elif i > 0 and k == 0 and i < (rowlen - 1):
                nbors = [
                    pixarr[i-1][k],
                    pixarr[i+1][k],
                    pixarr[i][k+1],
                    pixarr[i-1][k+1],
                    pixarr[i+1][k+1]
                ]
            elif i == (rowlen - 1) and k == 0:
                nbors = [
                    pixarr[i-1][k],
                    pixarr[i-1][k+1],
                    pixarr[i][k+1]
                ]
            elif i == (rowlen - 1) and k > 0 and k < (collen - 1):
                nbors = [
                    pixarr[i][k-1],
                    pixarr[i-1][k-1],
                    pixarr[i-1][k],
                    pixarr[i-1][k+1],
                    pixarr[i][k+1]
                ]
            elif i == (rowlen - 1) and k == (collen - 1):
                nbors = [
                    pixarr[i][k-1],
                    pixarr[i-1][k-1],
                    pixarr[i-1][k]
                ]

            borderPixel = False
            for val in nbors:
                if val == 0:
                    borderPixel = True
                    break

            if not borderPixel:
                continue

            perimeterCords.append([i, k])

    return perimeterCords


# calculates and returns C1 and C2 circularity
def getC1AndC2Circularity(perimeterCords, area, centroid):

    # Below perimeter length value is a close approximation.
    # We have N8 neighborhood and the objects with no overlapping perimeter
    # will have perimeter length same as number of boundry pixels. For the exact length we may need direction.
    perimeter = len(perimeterCords)
    C1 = (perimeter * perimeter) / area
    C2 = 0
    Cx = centroid[0]
    Cy = centroid[1]
    meanRD = 0
    standardDev = 0
    varianceRD = 0
    distList = []
    for point in perimeterCords:
        x = point[0]
        y = point[1]
        dist = math.sqrt(math.pow((x - Cx), 2) + math.pow((y - Cy), 2))
        distList.append(dist)
        meanRD += dist

    meanRD = meanRD / len(perimeterCords)

    for dist in distList:
        varianceRD += math.pow((dist - meanRD), 2)

    varianceRD = varianceRD / len(perimeterCords)
    standardDev = math.sqrt(varianceRD)
    C2 = meanRD / standardDev
    return [C1, C2]


# calculates row, column and mixed second moments.
# returns row moment, column moment, mixed moment
def getSecondMoments(perimeterCords, area, centroid):
    Cx = centroid[0]
    Cy = centroid[1]
    rowMoment = 0
    columnMoment = 0
    mixedMoment = 0

    for point in perimeterCords:
        x = point[0]
        y = point[1]

        rowv = x - Cx
        colnv = y - Cy
        mixedv = abs(rowv) * abs(colnv)

        rowMoment += math.pow(rowv, 2)
        columnMoment += math.pow(colnv, 2)
        mixedMoment += mixedv

    rowMoment = rowMoment / area
    columnMoment = columnMoment / area
    mixedMoment = mixedMoment / area
    return [rowMoment, columnMoment, mixedMoment]


# iteratively searches through the given object perimeter coordinates and returns the max, min useful
# for drawing bounding box around the region of interest
def getCornersForBoundingBox(perimeterCords, pixarray):
    rowlen = len(pixarray)
    collen = len(pixarray)

    maxCol = 0
    minCol = collen - 1
    maxRow = 0
    minRow = rowlen - 1

    for point in perimeterCords:
        r = point[0]
        c = point[1]
        if c > maxCol:
            maxCol = c
        if c < minCol:
            minCol = c
        if r > maxRow:
            maxRow = r
        if r < minRow:
            minRow = r

    return [maxRow, maxCol, minRow, minCol]


# just a quick utility method for displaying img
def showImg(img, title=""):
    plt.imshow(img)
    plt.title(title)
    plt.show()


# Union find data structure for storing label equivalencies
class Unionfind:

    # constructor method, start label is the starting color value of the label
    def __init__(self, startLabel):
        self.startLabel = startLabel
        self.parent = []
        self.size = []
        self.componentCount = 0

    # CCL process calls this method everytime a new label is created
    def pushNewLabel(self, label):
        # actual label value will change in the parent array for the ease of search.
        # start label = 0, startlabel + 1 = 1 ...
        self.parent.append(label - self.startLabel)
        self.size.append(1)
        self.componentCount += 1

    # find returns parent of the given label.
    # while searching for the parent, it reassigns each child directly to it's parent by compressing the search path
    def find(self, index):
        if index == self.parent[index]:
            return index

        self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    # merges two labels based on their size
    # Attaches small chain to the large one as child
    def merge(self, label1, label2):
        root1 = self.find(label1 - self.startLabel)
        root2 = self.find(label2 - self.startLabel)

        if root1 == root2:
            return

        if self.size[root1] < self.size[root2]:
            self.size[root2] += self.size[root1]
            self.parent[root1] = root2
        else:
            self.size[root1] += self.size[root2]
            self.parent[root2] = root1

        self.componentCount -= 1

    # returns total component count
    def getTotalComponents(self):
        return self.componentCount

    # returns the list of equivalencies
    def getLabelMap(self):
        return self.parent


print(""" *** Hello, This program reads gray-level images as input. It performs thresholding and removes noise by applying morphological filters, 
detects objects in the image by performing connected components analysis. It also calculates several features from the detected objects 
like Area, Perimeter, C1, C2 circularity, Second moments and Bounding box. ***\n """)

imgName = input("Enter name of the image file (accepted formats: .PGM(P2)):")

if not imgName.upper().endswith(".PGM"):
    print("only .PGM images are accepted")
    exit(0)

if not path.exists(imgName):
    print("file doesn't exist in the directory: " +
          os.path.dirname(os.path.realpath(__file__)))
    exit(0)

path = os.path.dirname(os.path.realpath(__file__)) + "\\" + imgName

imgFile = open(path, "r")

# image thresholding returns thresholded 2D array of image.
threshImg = threshold(imgFile, 180, "inverse_binary")

if threshImg == []:
    print("invalid image file")
    exit(0)

# showImg method uses matplotlib "imshow" method which requires image as 3D
threshImg = convert2dTo3d(threshImg)
showImg(threshImg, "After Thresholding")

# A 3 * 3 structuring element for closing
kernel = np.ones((3, 3), np.uint8)

# An OpenCV lib method for performing closing on the image
clsImg = cv.morphologyEx(threshImg, cv.MORPH_CLOSE, kernel)
showImg(clsImg, "Image after applying closing with a 3 * 3 structuring element")

pixarr = np.array(clsImg)

# converts 3D array returned to 2D
pixarr = pixarr[:, :, 0]

# custom CCL method
pixarr, startLabel, totComps, maxPixelVal, labelset, equivs = ConnectedComponentLabeling(
    pixarr)

showImg(convert2dTo3d(pixarr), "connected components: " + str(totComps))


# Every object in the image has certain label which is available in the labelset.
# Iterates over each label, extracting the region and calculating different features of the region in each iteration.
# For convenience of testing, bulk region and feature extration is avoided.
for lbl in labelset:
    region = extractRegion(lbl, pixarr)
    centroid, area = getCentroidAndArea(region)

    if area < 20:
        continue

    perimeterCords = getPerimeterPixels(region)
    C1, C2 = getC1AndC2Circularity(perimeterCords, area, centroid)

    rm, cm, mm = getSecondMoments(perimeterCords, area, centroid)
    maxRow, maxCol, minRow, minCol = getCornersForBoundingBox(
        perimeterCords, region)

    content = " Area: " + str(area) + "\n Centroid(Cx, Cy): " + \
        "(" + str('%.3f' % (centroid[0])) + \
        ", " + str('%.3f' % (centroid[1])) + ")" + \
        "\n C2 Circularity: " + str('%.3f' % (C2)) + \
        "\n Second Row Moment: " + \
        str('%.3f' % (rm)) + "\n Second Column Moment: " + str('%.3f' % (cm)) + \
        "\n Second Mixed Moment: " + \
        str('%.3f' % (mm)) + "\n Bounding Box drawn around the region"

    # current region features are flushed to the console
    print("features: " + str(lbl) + "\n" + content + "\n")
    start = (minCol, minRow)
    end = (maxCol, maxRow)

    region3d = convert2dTo3d(region)
    rectImg = cv.rectangle(region3d, start, end, (255, 255, 255), 1)
    plt.imshow(rectImg)

    # Coords need to changed based on the image size, below values are currently for image 5.
    plt.text(-400, 150, content, fontsize=15)
    plt.subplots_adjust(left=0.25)
    plt.show()
