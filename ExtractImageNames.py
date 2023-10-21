import os,re
infl=input("Please enter the name of the input text filefile (eg: ANN_7L_17N_2C_Prediction.txt):")
if infl =="":
    infl="ANN_7L_17N_2C_Prediction.txt"
# Open the input file
with open(infl, "r") as input_file:
    lines = input_file.readlines()

    # Extract the filename without the extension from the first line
    image_filename = os.path.splitext(os.path.basename(lines[0]))[0]

    # Extract the folder names from the first line
    path_parts = lines[0].split('/')
    folder_names = path_parts[:-1]

    # Print the folder names
    print("Folder names:")
    for name in folder_names:
        print(name)
    print()

    # Ask user for the start and end of the subfolders to be joined
    start_folder = input("Enter the start folder name: ")
    end_folder = input("Enter the end folder name: ")

    # Create the output file
    output_filename = "ExtractedImages_" + infl
    with open(output_filename, "w") as output_file:

        # Loop through the lines of the input file
        for line in lines:
            # Remove the newline character at the end of the line
            line = line.strip()

            # Skip the first line
            #if line == lines[0]:
            #    continue
            line_parts= re.split(r'\t|\s+', line)
            image_filename=str(line_parts[0].split('/')[-1])
            # Split the path using '/'
            path_parts = line.split('/')
            # Find the indices of the start and end folders
            start_index = path_parts.index(start_folder)
            end_index = path_parts.index(end_folder)

            # Join the subfolders between the start and end indices
            subfolders = '/'.join(path_parts[start_index:end_index+1])
           # Remove the extension from the image filename
            #image_name = os.path.splitext(image_filename)[0]
            image_name =image_filename.replace('.pkl', '')
            # Create the output line by joining the subfolders with the image filename and adding the '.pkl' extension
            output_line = os.path.join(subfolders, image_name) +'\t' + '\t'.join(line_parts[1:]) + "\n"

            # Write the output line to the output file
            output_file.write(output_line)

print(f"Output written to {output_filename}")
show_img=input("Do you wnt the images to be displayed now? (yes):")
if show_img =="yes":
    from PIL import Image
    from pathlib import Path
    import tkinter as tk
    from PIL import Image, ImageTk
    global index, tk_images,image

    # create tkinter window
    window = tk.Tk()

    # create a canvas to display images
    canvas = tk.Canvas(window, width=500, height=500)
    canvas.pack()

    # read image paths and titles from file
    with open(output_filename, 'r') as file:
        lines = file.readlines()

    # create empty lists to store images and their titles
    images = []
    titles = []

    # iterate through each line in the file
    for line in lines:
        # split the line into image path and title
        image_path = line.strip().split('\t')
        title=str('\t'.join(image_path[0:]))
        # Read the image from the file using Pillow
        # open the image file using Pillow
        image = Image.open(image_path[0])

        # add the image and title to their respective lists
        images.append(image)
        titles.append(title)

    # create a list of PhotoImage objects
    tk_images = [ImageTk.PhotoImage(image) for image in images]

    # create a label to display the images
    label = tk.Label(canvas, text='', compound='top')
    label.pack()

    # set the initial image and title
    index = 0
    label.config(image=tk_images[index], text=titles[index])

    # function to display next image
    def next_image(event):
        global index
        index += 1
        if index >= len(tk_images):
            index = 0
        label.config(image=tk_images[index], text=titles[index])
        print(titles[index])

    # function to display previous image
    def prev_image(event):
        global index
        index -= 1
        if index < 0:
            index = len(tk_images) - 1
        label.config(image=tk_images[index], text=titles[index])
        print(titles[index])

    # bind arrow keys to next_image and prev_image functions
    window.bind('<Up>', prev_image)
    window.bind('<Down>', next_image)

    # function to scale the image to match window size when the window is resized
    def scale_image(event):
        global index, tk_images, images, label
        # get the current image
        image = images[index]
        # get the current window size
        w = event.width
        h = event.height
        # calculate the aspect ratio of the image
        aspect_ratio = image.width / image.height
        # calculate the new size of the image while maintaining the aspect ratio
        if w / h > aspect_ratio:
            new_w = int(h * aspect_ratio)
            new_h = h
        else:
            new_w = w
            new_h = int(w / aspect_ratio)
        # resize the image to match the new size
        image = image.resize((new_w, new_h))
        # update the PhotoImage object
        tk_images[index] = ImageTk.PhotoImage(image)
        # update the label with the scaled image
        label.config(image=tk_images[index], text=titles[index])
    def scale_image(event):
        global index, tk_images, images, label, manual_resize
        # check if the window size was changed manually by the user
        if manual_resize:
            # get the current image
            image = images[index]
            # get the current window size
            w = event.width
            h = event.height
            # calculate the aspect ratio of the image
            aspect_ratio = image.width / image.height
            # calculate the new size of the image while maintaining the aspect ratio
            if w / h > aspect_ratio:
                new_w = int(h * aspect_ratio)
                new_h = h
            else:
                new_w = w
                new_h = int(w / aspect_ratio)
            # resize the image to match the new size
            image = image.resize((new_w, new_h))
            # update the PhotoImage object
            tk_images[index] = ImageTk.PhotoImage(image)
            # update the label with the scaled image
            label.config(image=tk_images[index], text=titles[index])
        # set the flag to False after the first automatic resize event
        manual_resize = True

    # set the initial flag value to False
    manual_resize = False

    # bind the scale_image function to the window resize event
    window.bind('<Configure>', scale_image)
    # bind the scale_image function to the window resize event
    window.bind('<Configure>', scale_image)
    # function to quit when 'q' key is pressed
    def quit(event):
        window.quit()
        window.destroy()

    # bind 'q' key to the quit function
    window.bind('q', quit)

    # run the tkinter event loop
    window.mainloop()
