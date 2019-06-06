
import numpy as np
import cv2 as cv
import random


def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out


def computeEnergyFunctionGradient(img, ref_img, smoothness_factor, coord_x, coord_y, neighborhood):
   return (ref_img[coord_x, coord_y] + (smoothness_factor * float(sum(neighborhood))))/(1 + (smoothness_factor * float(len(neighborhood))))


def getNeighborhood(img, coord_x, coord_y):
     neighborhood = list()
     neighborhood.append(img[coord_x, coord_y + 1]) 
     neighborhood.append(img[coord_x - 1,coord_y])
     neighborhood.append(img[coord_x + 1,coord_y])
     neighborhood.append(img[coord_x, coord_y - 1]) 
     return neighborhood


def solveByGaussSeidel(img, smoothness_factor, epochs):
    print('\n\nPreparing Gauss-Seidel Method...')
    print("\nSmoothness factor = ",smoothness_factor)
    print("\nNeighborhood size = ", 4)
    print("\nEpochs = ", epochs)
    input("\n\nPress Enter to continue...")
    print('\n-------------------------------------------------------------')
    print('\n\n------------Running Gauss-Seidel Iterations----------------')
    print('\n\n-----------------------------------------------------------')
    img_width = img.shape[0]
    img_height = img.shape[1]
    ref_img = img
    for i in range(0, epochs):
        for coord_x in range(1, img_width - 1):
                for coord_y in range(1, img_height - 1):
                        neighborhood = getNeighborhood(img, coord_x, coord_y)
                        img[coord_x, coord_y] = computeEnergyFunctionGradient(img, ref_img, smoothness_factor, coord_x, coord_y, neighborhood)
        img = cv.normalize(img.astype('float'), None, 0, 1, cv.NORM_MINMAX)
    return img  


def sampleImage(num_samples, img):
    print('\n\n')
    for i in range(0, num_samples):     
        coordX = random.randint(0, img.shape[0] - 1)
        coordY = random.randint(0,img.shape[1] - 1)
        print('Sample ', i, ' = ',img[coordX, coordY])



def main():
    noisy_img = cv.imread('lena_noise.png',0)
    noisy_img = cv.normalize(noisy_img.astype('float'), None, 0, 1, cv.NORM_MINMAX)
    smoothness_factor = float(input('Especifique un factor de suavidad\n'))
    epochs = int(input('Especifique un numero de epocas...\n'))
    filtered_img = solveByGaussSeidel(noisy_img, smoothness_factor, epochs)
    print('final sample')
    sampleImage(5, filtered_img)
    cv.imshow('a', filtered_img)
    cv.waitKey(0)


main()