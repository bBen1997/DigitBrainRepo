import cv2
import numpy as np
import random
import operator
#Variable for filtering noise lines
minLineLength=5

#Find min max points in contour
def findMinMaxXPointsInContour(contour):
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])

    
    return [leftmost, rightmost]


#Detect overlap defect regions on image
def detectOverlapsOnImage(inputImage):
    resultOverlapsRect = []
    #Read image
    nparr = np.frombuffer(inputImage, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
           
    imageInitState = image.copy()
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #edged = cv2.Canny(gray, 120, 150)    
    #lines = cv2.HoughLinesP(image=edged,rho=1,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=minLineLength,maxLineGap=200)
    #a,b,c = lines.shape
    #for i in range(a):
    #    cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3, cv2.LINE_AA)
    #contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #image_copy = image.copy()
    #cv2.drawContours(image_copy, contours, -1, (0,255,0), 3)
    image_res = image.copy()

   #Check if current pixel is green and belongs to overlap and create mask    
    for y in range(height):
        for x in range(width):
        # Get the color of the pixel
            if image[y, x][0] < 10 and  image[y, x][1] > 120 and image[y, x][1] < 130  and  image[y, x][2] < 10:
                image_res[y,x] = [255,255,255]
            else:
                image_res[y,x] = [0,0,0]

    #cv2.imshow("IRes", image_res)
    cv2.waitKey(1)
       
    #Find overlap region on mask with contour analyze
    image_res_gray = cv2.cvtColor(image_res,cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image_res_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        [xMin,xMax] = findMinMaxXPointsInContour(contour)
      
        if w > minLineLength and h > minLineLength:
           
           # imageInitState =  cv2.circle(imageInitState, xMin, 5, (0,255,0), -1)
           # imageInitState =  cv2.circle(imageInitState, xMax, 5, (0,255,0), -1)             
            imageInitState = cv2.rectangle(imageInitState,(x,y),(x+w,y+h),(0,255,0),5)
            widthTmp = tuple(map(operator.sub, xMax, xMin))          
            resultOverlapsRect.append([x,y,widthTmp[0],h])
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))    
        # Draw the contour on the original image
        cv2.drawContours(image_res, [contour], 0, color, 5)

    #Show result images 
    # cv2.imshow("FinalRes", imageInitState) 
    # cv2.imshow("Window",image_res_gray)
    # cv2.imshow("Window1",image_res)
    cv2.waitKey(10)
    return resultOverlapsRect



#Check if two rectangles are intersecting


def rectAreIntersecting(rect1, rect2):
    topRightX1 = rect1[0] + rect1[2]
    topRightY1 = rect1[1]
    
    topRightX2 = rect2[0] + rect2[2]
    topRightY2 = rect2[1]
    
    
    
    bottomLeftX1 =  rect1[0] 
    bottomLeftY1 =  rect1[1] + rect1[3]
    
    bottomLeftX2 =  rect2[0] 
    bottomLeftY2 =  rect2[1] + rect2[3]
    
    return not ( topRightX1 < bottomLeftX2 or bottomLeftX1 > topRightX2 or topRightY1 < bottomLeftY2 or  bottomLeftY1 >  topRightY2)


def intersectsBox(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])



#Function for gaps detection
def detectGapsOnImage(inputImage):
    resultGapsRect = []
    #Read image
    nparr = np.frombuffer(inputImage, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    height, width, channels = image.shape
    imageInitState = image.copy()
    #Convert grayscale                
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Edge detection
    edged = cv2.Canny(gray, 120, 150)
    #Find horizontal and vertical lines on edge detected image     
    lines = cv2.HoughLinesP(image=edged,rho=1,theta=np.pi/180, threshold=200,lines=np.array([]), minLineLength=minLineLength,maxLineGap=200)
    a,b,c = lines.shape
    #Draw lines
    for i in range(a):
        cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 255, 0), 3, cv2.LINE_AA)
    #Make contour detection on image
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, (0,255,0), 3)
    image_res = image.copy()
    #Create a gap mas on image
    for y in range(height):
        for x in range(width):
        # Get the color of the pixel
            if image[y, x][0] == image_copy[y,x][0] and  image[y, x][1] == image_copy[y,x][1] and  image[y, x][2] == image_copy[y,x][2] :
                image_res[y,x] = [0,0,0]
            else:
                image_res[y,x] = [255,255,255]
    #Find mask regions and draw founding boxes                
    image_res = cv2.medianBlur(image_res,7)
    image_res_gray = cv2.cvtColor(image_res,cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(image_res_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours):
        x,y,w,h = cv2.boundingRect(contour)
        [xMin,xMax] = findMinMaxXPointsInContour(contour)
      
        if w > 10 and h > 5:
                     
            imageInitState = cv2.rectangle(imageInitState,(x,y),(x+w,y+h),(0,255,0),5)
            widthTmp = tuple(map(operator.sub, xMax, xMin))          
            resultGapsRect.append([x,y,widthTmp[0],h])
        # Generate a random color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
        # Draw the contour on the original image
        cv2.drawContours(image_res, [contour], 0, color, 5)


     #visualize images and return result
    # cv2.imshow("FinalRes", imageInitState)
    # cv2.imshow("Window",image_res_gray)
    # cv2.imshow("Window1",image_res)
    cv2.waitKey(10)
    return resultGapsRect


#Check if two masks are overlaping
def do_overlap(box1, box2):
     
    # if rectangle has area 0, no overlap
  
    if box1[0] == box1[0]+ box1[2] or box1[1] == box1[1] + box1[3] or box2[0] == box2[0]+ box2[2] or box2[1] == box2[1] + box2[3]:
        return False
     
    # If one rectangle is on left side of other
    print(box1[0])
    print(box2[0] + box2[2])
    if box1[0] >= box2[0]  + box2[2] or box2[0]>= box1[0] + box1[2]:
        return False
   
    print("!!!!!!!!!!!!")
    # If one rectangle is above other
    if box1[1]  >= box2[1]  + box2[3]  or box2[1] >= box1[1] + box1[3]:
        return False
    print("True OVerlap")
    return True


#Main function, call gaps and overlap detection
def main(inputImage):
       
    #inputImagePath = sys.argv[1]         
    resultGapsRect = detectGapsOnImage(inputImage)
    resultOverlapsRect = detectOverlapsOnImage(inputImage)
    #pathname, extension = os.path.splitext(inputImagePath)
    # resImageFilePath = pathname + "_res" + extension
    # resTxtFilePathOver = pathname + "_overlap.txt"
    # resTxtFilePathGaps = pathname + "_gaps.txt"
    resTxtFilePath = "DefectResult" + ".txt"

    resultString = ""
    #Filter results and write into txt files and output images.     
    resultGapsRectFilt = []
    for i in range(len(resultGapsRect)):
        print("Start gaps vector ")
        isFoundInterRect = False    
        for j in range(len(resultOverlapsRect)):
            box1 = [resultGapsRect[i][0], resultGapsRect[i][1], resultGapsRect[i][0] + resultGapsRect[i][2], resultGapsRect[i][1]+ resultGapsRect[i][3]]
            box2 = [resultOverlapsRect[j][0], resultOverlapsRect[j][1], resultOverlapsRect[j][0] + resultOverlapsRect[j][2], resultOverlapsRect[j][1]+ resultOverlapsRect[j][3]]
            if   (do_overlap(resultGapsRect[i], resultOverlapsRect[j])):
                print("intersecint")
                isFoundInterRect = True                
                break
        if  not(isFoundInterRect):               
            resultGapsRectFilt.append(resultGapsRect[i])

    for i in range(len(resultOverlapsRect)):
        #imgRes = cv2.rectangle(imgRes,(resultOverlapsRect[i][0],resultOverlapsRect[i][1]),(resultOverlapsRect[i][0]+resultOverlapsRect[i][2],resultOverlapsRect[i][1]+resultOverlapsRect[i][3]),(0,255,0),5)            
        finalStrForOutput = '[O,' + str(resultOverlapsRect[i][0]) + ',' + str(resultOverlapsRect[i][1]) + ',' + str(resultOverlapsRect[i][2]) + ',' + str(resultOverlapsRect[i][3]) + ']' + ' '
        resultString += finalStrForOutput 


    
    for i in range(len(resultGapsRectFilt)):   
        #imgRes = cv2.rectangle(imgRes,(resultGapsRectFilt[i][0],resultGapsRectFilt[i][1]),(resultGapsRectFilt[i][0]+resultGapsRectFilt[i][2],resultGapsRectFilt[i][1]+resultGapsRectFilt[i][3]),(0,255,255),5)                     
        finalStrForOutput = '[G,' + str(resultGapsRectFilt[i][0]) + ',' + str(resultGapsRectFilt[i][1]) + ',' + str(resultGapsRectFilt[i][2]) + ',' + str(resultGapsRectFilt[i][3]) + ']' + ' '
        resultString += finalStrForOutput 

    return resultString;
# if __name__ == '__main__':

#     if len(sys.argv) != 2:
#         print("PLease call script with input image path")
#         exit()
#     main(sys.argv[1:])
   














#print("Before exit")
#exit()
#print("After exit ")
# define a (3, 3) structuring element
#kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# apply the dilation operation to the edged image
#dilate = cv2.dilate(edged, kernel, iterations=1)

# find the contours in the dilated image
#contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#image_copy = image.copy()
# draw the contours on a copy of the original image
#for cont in contours:
  #  print("Curr Conpouts")
#cv2.drawContours(image_copy, contours, -1, (0,255,0), 5)
#print(len(contours), "objects were found in this image.")

#cv2.imshow("Dilated image", dilate)
#cv2.imshow("contours", image_copy)
#cv2.waitKey(0)
