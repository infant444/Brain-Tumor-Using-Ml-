import cv2

img=cv2.imread("result/11.png", cv2.IMREAD_GRAYSCALE)
img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

# img=cv2.applyColorMap(img,cv2.COLORMAP_JET)

cv2.imwrite("x.png",img)