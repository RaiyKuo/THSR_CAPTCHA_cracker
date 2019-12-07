import numpy as np
import matplotlib.pyplot as plt                                  #顯示圖片
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import cv2


def showImage(img1, img2 = None, save_name = False):
    if img2 is not None:
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
    else:
        plt.imshow(img1)
    if save_name:
        plt.savefig('preprocess/log_vs_pow/' + save_name + '.png')
    else:
        plt.show()


def curveRegPow(x, y, shift=130, pow = 3):
    X, Y = 1/(x+shift)**pow, y
    poly_reg = PolynomialFeatures(degree=1)
    X_ = poly_reg.fit_transform(X.T)
    lr = LinearRegression()
    lr.fit(X_, Y)

    x2 = np.array([[i for i in range(0, width)]])
    y2 = lr.predict(poly_reg.fit_transform(1/(x2.T+shift)**pow))
    return (x2, y2)


def curveRegLog(x, y, shift=10):
    X, Y = np.log(x + shift), y
    poly_reg = PolynomialFeatures(degree=1)  # 先轉成 y = m*ln(x) + b 後線性回歸
    X_ = poly_reg.fit_transform(X.T)
    lr = LinearRegression()
    lr.fit(X_, Y)

    x2 = np.array([[i for i in range(0, width)]])
    y2 = lr.predict(poly_reg.fit_transform(np.log(x2 + shift).T))
    return (x2, y2)


def eraseCurveFromImg(img_in, x2, y2):
    for x, y in np.column_stack([y2.round(0), x2[0]]):
        pos = height - int(x)
        r = 3
        img_in[pos-r:pos+r, int(y)] = 255 - img_in[pos-r:pos+r, int(y)]
    return img_in


num_of_img = 5

for n in range(1, num_of_img + 1):
    n = str(n)
    img = cv2.imread('demo/' + n +'.png')
    dst = cv2.fastNlMeansDenoising(img, None, 51, 3, 21)               # Remove background noise
    #  showImage(img, dst)

    ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY_INV)  # Invert scale
    #  showImage(thresh)

    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)                  # Turn colors into black and white

    height, width = imgarr.shape[0], imgarr.shape[1]
    img_hollow = imgarr.copy()
    img_hollow[:, 13: width-6] = 0                          # For removing the noise arc
    #  showImage(img_hollow)

    axis_y, axis_x = np.where(img_hollow == 255)            # Find the residual part of the noise arc
    x, y = np.array([axis_x]), height - axis_y

    POW = curveRegPow(x, y)                                 # Regression the residual part to obtain the curve formula
    img_final = eraseCurveFromImg(imgarr, POW[0], POW[1])   # Remove the curve based on formula
    #  showImage(img, img_final)

    kernel = np.ones((2, 4), np.uint8)
    img_final = cv2.morphologyEx(img_final, cv2.MORPH_OPEN, kernel)  # Remove noise further by Erosion and Dilution
    #  showImage(img, img_final)

    dim = (140, 48)
    img_final = cv2.resize(img_final, dim, interpolation=cv2.INTER_AREA)  # Reszie

    img_final = cv2.cvtColor(img_final, cv2.COLOR_GRAY2RGB)  # Turn grayscale back to RGB 3 channel for requirement of training

    cv2.imwrite('demo/preprocessed_img/' + n + '.jpg', img_final)  # Output the image
    print('{}.jpg'.format(n))

