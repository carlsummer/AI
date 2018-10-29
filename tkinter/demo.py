
from tkinter import *


def main():
    root = Tk()

    w = Canvas(
        root,
        width=200,
        height=200,
        background="white"
    )
    w.pack()

    w.create_line(0, 100, 200, 100, fill='yellow')

    w.create_line(100, 0, 100, 200, fill='red', dash=(4, 4))

    w.create_rectangle(50, 50, 150, 150, fill='blue')

    mainloop()


if __name__ == '__main__':
    main()