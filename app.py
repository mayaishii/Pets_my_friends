from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter import messagebox as mb

import pandas as pd
from PIL import Image, ImageTk
from predictor import Predictor

from tensorflow.keras.preprocessing import image as keras_image

import tkinter as tk

LARGEFONT = ('Grandview', 35)
MEDIUMFONT = ('Grandview', 25)
SMALLFONT = ('Grandview', 17)
FONT14 = ('Grandview', 14)
EMBEDDINGS_PATH = "image_embeddings.csv"
META_DATA_PATH = "animals_data.csv"
IMAGES_PATH = "photos/"


class tkinterApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.geometry("800x600")
        self.title("MyFriendSearch")
        self.resizable(width=False, height=False)

        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        frame = SearchPage(container, self)
        self.frames[SearchPage] = frame
        frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(SearchPage)

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()


class SearchPage(tk.Frame):

    def openfn(self):
        filename = fd.askopenfilename(title='open')
        return filename

    def open_img(self):
        fn = self.openfn()
        img = Image.open(fn)
        w, h = img.size
        nh, nw = self.cals_new_size(w, h, 200, 200)
        resize_img = img.resize((nw, nh))
        photo = ImageTk.PhotoImage(resize_img)
        self.image = img
        self.image_obj = self.cat_image_canvas.create_image(100, 100, image=photo, anchor=CENTER)

        self.cat_image_canvas.image = photo
        self.current_image_path = fn

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent, height = 500, width = 500)

        self.title = ttk.Label(self, text="MyFriendSearch", font=LARGEFONT)
        self.title.grid(row=0, column=1, padx=10, pady=10)

        self.result_lable = ttk.Label(self, text="Результаты поиска", font=MEDIUMFONT, background = "white")
        self.result_lable.place(x=350, y=30)

        self.cat_image_canvas = Canvas(self, width=200, height=200, bg='white')
        self.cat_image_canvas.grid(row=5, column=0, columnspan=8, padx=5, pady=5)


        load_button = ttk.Button(self, text="Загрузить фотографию", command=lambda: self.open_img())
        load_button.grid(row=1, column=1, padx=10, pady=10)

        search_button = ttk.Button(self, text="Начать поиск", command=lambda: self.find_similar())
        search_button.grid(row=12, column=1, padx=10, pady=10)

        self.predictor = Predictor(embeddings_path=EMBEDDINGS_PATH, images_path=IMAGES_PATH)
        self.meta_data = pd.read_csv(META_DATA_PATH)
        self.current_image_path = None

        self.n_similar = 3
        self.similar_images_objects = []
        self.feed_y = 75
        self.feed_x = 350

        self.y_cord = []
        for image_number in range(self.n_similar):
            image_canvas = Canvas(self, width=150, height=150, bg='white')
            image_canvas.create_rectangle(3, 3, 150, 150, outline='white')
            image_canvas.place(x=self.feed_x, y=self.feed_y)
            self.similar_images_objects.append(image_canvas)
            self.y_cord.append(self.feed_y)
            self.feed_y = self.feed_y + 175

        self.meta_info = []

    def cals_new_size(self, w, h, nh, nw):
        if h > w:
            nh = 200
            nw = int(w * nh / h)
        else:
            nw = 200
            nh = int(h * nw / w)
        return nh, nw

    def find_similar(self):
        if not self.current_image_path:
            mb.showwarning(title="", message="Сначала загрузите фото")
            return 0
        img = keras_image.load_img(self.current_image_path, target_size=(224, 224))
        self.similar_images = self.predictor.find_similar_images(img)
        self.show_similar()

    def show_similar(self):
        if len(self.meta_info) > 0:
            for entry in self.meta_info:
                entry['gender_lable'].destroy()
                entry['age_lable'].destroy()
                entry['number_lable'].destroy()
        self.meta_info = []

        for idx in range(self.n_similar):
            image_path = IMAGES_PATH + self.similar_images[idx]
            img = Image.open(image_path)
            w, h = img.size
            nh, nw = self.cals_new_size(w, h, 200, 200)
            resize_img = img.resize((nw, nh))
            photo = ImageTk.PhotoImage(resize_img)

            self.similar_images_objects[idx].create_image(86, 86, image=photo, anchor=CENTER)
            self.similar_images_objects[idx].image = photo

            _, gender, age, number = self.meta_data[self.meta_data.image == self.similar_images[idx]].values[0]

            gender_lable = ttk.Label(self, text=gender, font=SMALLFONT, background = "white")
            gender_lable.place(x = self.feed_x + 170, y = self.y_cord[idx] + 10)

            age_lable = ttk.Label(self, text=age, font=FONT14, background = "white")
            age_lable.place(x=self.feed_x + 170, y=self.y_cord[idx] + 30)

            number_lable = ttk.Label(self, text=number, font=FONT14)
            number_lable.place(x=self.feed_x + 170, y=self.y_cord[idx] + 70)

            self.meta_info.append({"gender_lable": gender_lable, 'age_lable': age_lable, "number_lable": number_lable})

app = tkinterApp()
app.mainloop()

