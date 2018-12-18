from tkinter import *
import random
from PIL import Image, ImageDraw, ImageGrab, PngImagePlugin
from NeuralNetProd import NeuralNetwork

output = None

def save():
    filename = "image.png"
    image1.save(filename)

def xy(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    global lastx, lasty
    canvas.create_line((lastx, lasty, event.x, event.y), fill="white", width=1)
    lastx, lasty = event.x, event.y

def getter():
    x2=leftframe.winfo_rootx()+canvas.winfo_x()
    y2=leftframe.winfo_rooty()+canvas.winfo_y()
    x1=x2+canvas.winfo_width()
    y1=y2+canvas.winfo_height()
    print("save")
    ImageGrab.grab().crop((x2,y2,x1,y1)).save("./test.jpg", optimize=True, quality=100)

    output = NeuralNetwork()
    outputnum = Label(rightframe, text="The computer thinks your number is: " +  str(output))
    outputnum.pack()


root = Tk()
root.title("Computer Guesses Number")
root.geometry = ("428, 428")
lastx, lasty = 0, 0

frame = Frame(root, height=428, width=428)
frame.pack()
leftframe = Frame(frame, height=400, width=400)
leftframe.pack(side=LEFT)
rightframe = Frame(frame, height=28, width=28)
rightframe.pack(side=LEFT)

canvash = 400
canvasw = 400
canvas = Canvas(leftframe, width=canvasw, height=canvash)
canvas.configure(background="black", highlightthickness=2)
canvas.pack(side=LEFT)
canvas.bind("<Button-1>", xy)
image1 = canvas.bind("<B1-Motion>", addLine)

button1 = Button(frame, text="Submit",command=lambda:getter())
button1.pack(side=BOTTOM)

root.mainloop()