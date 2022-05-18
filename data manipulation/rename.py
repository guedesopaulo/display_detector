import os
folder = "frames"
for count, filename in enumerate(os.listdir(folder)):
    dst = f"{str(count)}.jpg"
    src =f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
    dst =f"{folder}/{dst}"
         
    # rename() function will
    # rename all the files
    os.rename(src, dst)
