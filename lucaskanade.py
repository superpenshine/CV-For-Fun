import cv2
import os
import numpy as np
from scipy import signal

def show(img):
    cv2.imshow('abc', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1] - 1);
    x1 = np.clip(x1, 0, im.shape[1] - 1);
    y0 = np.clip(y0, 0, im.shape[0] - 1);
    y1 = np.clip(y1, 0, im.shape[0] - 1);

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


'''
yx, xy: window meshgrid
f_s: feature scaled 
g: previous guess
v: tmp guess of iteration k
x_t, y_t: (X + g) + v
'''
def PyrLK(frame1, frame2, features, path=None, filename=None, 
    corners=100, window=3, minEigenValue=1e-3, maxLevel=3):

    features = features.reshape(-1, 2)
    color = np.random.randint(0, 255, (100,3)) 
    halfWin = int(window/2)
    yx, xy = np.meshgrid(np.asarray(range(-halfWin, halfWin+1)), np.asarray(range(-halfWin, halfWin+1)))
    pyr1, pyr2 = [frame1], [frame2]
    for i in range(maxLevel):
        pyr1.append(cv2.pyrDown(pyr1[-1]))
        pyr2.append(cv2.pyrDown(pyr2[-1]))

    def lucas_kanade(level, guessPrev):
        featuresScaled = features / 2**level
        curr, next = pyr1[level], pyr2[level]
        h, w = curr.shape
        fx = signal.convolve2d(curr, np.array([[-1, 1], [-1, 1]]) , mode='same')
        fy = signal.convolve2d(curr, np.array([[-1, -1], [1, 1]]) , mode='same')
        blur = cv2.GaussianBlur(curr, (window, window), 0)
        guess, state = [], []

        for f_s, g in zip(featuresScaled, guessPrev):
            x_s, y_s = f_s.ravel()
            x_t, y_t = x_s + g[0], y_s + g[1]

            # Check if point out of img
            if x_t < 0 or y_t < 0 or x_t > w - 1 or y_t > h - 1:
                guess.append(np.array([0.0, 0.0]))
                state.append(0)
                continue

            yx_s, xy_s = (yx + x_s).ravel(), (xy + y_s).ravel()
            Ix = bilinear_interpolate(fx, yx_s, xy_s)
            Iy = bilinear_interpolate(fy, yx_s, xy_s)
            A_t = np.array(np.matrix((Ix, Iy)))
            structure_tensor = np.dot(A_t, A_t.T)

            # Iterasive lk
            v = np.array([0.0, 0.0])
            k = 3
            healthy = True
            while k:
                k -= 1
                x_t, y_t = f_s.ravel() + g + v
                yx_t, xy_t = (yx + x_t).ravel(), (xy + y_t).ravel()
                It = bilinear_interpolate(curr, yx_s, xy_s) - bilinear_interpolate(next, yx_t, xy_t)

                # Check if structure tensor pinv is ill conditioned (small eigen value)
                U, S, V_t = np.linalg.svd(structure_tensor, hermitian=True)
                if np.min(abs(S)) >= minEigenValue:
                    v += V_t.T.dot(np.diag(1 / S)).dot(U.T).dot(A_t).dot(-It)
                else:
                    healthy = False
                    break

            if healthy:
                guess.append(g + v)
                state.append(1)
            else:
                guess.append(np.array([0.0, 0.0]))
                state.append(0) 

        if level == 0:
            return (np.array(guess) + features).reshape(-1, 1, 2), np.array(state).reshape((-1, 1)), None

        return lucas_kanade(level - 1, 2 * np.array(guess))

    return lucas_kanade(maxLevel, np.zeros((len(features), 2)))
