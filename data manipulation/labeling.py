import cv2, numpy as np
import pandas as pd


#reading .csv, it need to be have the fallowing colunms: name;x1;y1;x2;y2;x3;y3;x4;y4
df = pd.read_csv("vai.csv")

# Mouse callback function to get x,y points
global click_list
positions, click_list = [], []
def callback(event, x, y, flags, param):
    if event == 1:
        click_list.append(x)
        click_list.append(y)


"""
WAY TO USE:
    mouse_click = put point
    f = fullscreen
    e = erase point
    backspace = go to previus image
    space = go to next image (if have 4 points in the screen)
    t = fine tuning
    w a s d = fine tuning controll
    esq = quit
"""

fullScreen = False
fine_tuning  = False
cv2.namedWindow("img", cv2.WINDOW_NORMAL)
cv2.setMouseCallback('img', callback)

#image number to be read
j = 0

# Mainloop
while True:
    font = cv2.FONT_HERSHEY_SIMPLEX
    name = "dataset/train/" + str(j) + ".jpg"
    save = str(j) + ".jpg"
    img = cv2.imread(name)
    
    #print points in the image
    if len(click_list) !=0:
        for i in range(int(len(click_list)/2)):  
            cv2.putText(img, '.',(click_list[i*2],click_list[i*2+1]),font,
                    1, (255, 0, 0), 1) #last number controll the radius of the point    
            
    cv2.imshow('img', img)
    # Wait, and allow the user to quit with the 'esc' key
    k = cv2.waitKey(1)
    # If user presses 'esc' break 
    if k == 27: break 

    if k == 32:
        if len(click_list) == 8:
            click_list.insert(0,save)
            if j == len(df):
                df.loc[len(df)] = click_list
            else:
                df["name"][j] = click_list[0]
                df["x1"][j] = click_list[1]
                df["y1"][j] = click_list[2]
                df["x2"][j] = click_list[3]
                df["y2"][j] = click_list[4]
                df["x3"][j] = click_list[5]
                df["y3"][j] = click_list[6]
                df["x4"][j] = click_list[7]
                df["y4"][j] = click_list[8]
            click_list = []
            j = j+1
            df.to_csv('vai.csv',index=False)
            
    
    if k == ord("e"):
        if len(click_list) !=0:
            click_list.pop()
            click_list.pop()    
    if k == 8:
        j = j -1
            
    if k == ord("f"):
        if not fullScreen:
            cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty("img", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        fullScreen = not fullScreen

    if k == ord("t"):
        fine_tuning = not fine_tuning

    if fine_tuning and k == ord("d"):
        i = len(click_list)
        click_list[i-2] =  click_list[i-2] + 1
        #print(click_list)
    if fine_tuning and k == ord("a"):
        i = len(click_list)
        click_list[i-2] =  click_list[i-2] - 1

    if fine_tuning and k == ord("w"):
        i = len(click_list)
        click_list[i-1] =  click_list[i-1] - 1 

    if fine_tuning and k == ord("s"):
        i = len(click_list)
        click_list[i-1] =  click_list[i-1] + 1 
             
cv2.destroyAllWindows()
    
  

