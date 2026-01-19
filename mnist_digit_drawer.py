"""
Application Tkinter pour dessiner des chiffres et obtenir des prédictions MNIST
"""

import tkinter as tk
from tkinter import Canvas, Button, Label
from PIL import Image, ImageDraw
import numpy as np
from tensorflow import keras


# Chargez le modèle sauvegardé
loaded_model = keras.models.load_model('mnist_cnn_model.h5')
 
def predict_digit(image_array, model):
    """Prédit le chiffre dans une image"""
    if image_array.max() > 1.0:
        image_array = image_array.astype('float32') / 255.0
    
    if len(image_array.shape) == 2:
        image_array = np.expand_dims(image_array, axis=-1)
    
    image_array = np.expand_dims(image_array, axis=0)
    probabilities = model.predict(image_array, verbose=0)
    prediction = np.argmax(probabilities)
    
    return prediction, probabilities[0]


class DigitDrawer:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.root.title("Dessinez un chiffre")
        self.root.geometry("400x500")
        
        Label(root, text="Dessinez un chiffre (0-9)", font=('Arial', 18, 'bold')).pack(pady=10)
        
        self.canvas = Canvas(root, width=280, height=280, bg='white', cursor='cross')
        self.canvas.pack(pady=20)
        
        # Créer une image PIL pour le dessin
        self.image = Image.new('L', (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        self.last_x = None
        self.last_y = None
        self.canvas.bind('<B1-Motion>', self.draw_on_canvas)
        self.canvas.bind('<ButtonPress-1>', self.start_draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        
        Button(root, text='Prédire', command=self.predict_digit, 
               font=('Arial', 14), bg='#4CAF50', fg='white', padx=30, pady=10,
               state='normal' if self.model else 'disabled').pack(pady=10)
        
        Button(root, text='Effacer', command=self.clear_canvas,
               font=('Arial', 14), bg='#f44336', fg='white', padx=30, pady=10).pack(pady=5)
        
        self.prediction_label = Label(root, text='Dessinez un chiffre',
                                     font=('Arial', 14), wraplength=350)
        self.prediction_label.pack(pady=20)
        
        self.prob_label = Label(root, text='', font=('Arial', 10), wraplength=350)
        self.prob_label.pack(pady=5)
    
    def start_draw(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_draw(self, _event):
        self.last_x = None
        self.last_y = None
    
    def draw_on_canvas(self, event):
        if self.last_x and self.last_y:
            # Dessiner sur le canvas Tkinter
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                    width=15, fill='black', capstyle=tk.ROUND, smooth=tk.TRUE)
            # Dessiner sur l'image PIL
            self.draw.line([self.last_x, self.last_y, event.x, event.y], fill=0, width=15)
        self.last_x = event.x
        self.last_y = event.y
    
    def clear_canvas(self):
        self.canvas.delete('all')
        # Réinitialiser l'image PIL
        self.image = Image.new('L', (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text='Dessinez un chiffre')
        self.prob_label.config(text='')
    
    def predict_digit(self):
        if self.model is None:
            self.prediction_label.config(text='Modèle non chargé')
            return
        
        # Redimensionner l'image à 28x28
        img = self.image.resize((28, 28), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.Resampling.LANCZOS)
        
        # Convertir en array numpy et inverser (blanc->noir, noir->blanc)
        img_array = np.array(img)
        img_array = 255 - img_array
        img_array = img_array.astype('float32') / 255.0
        
        prediction, probabilities = predict_digit(img_array, self.model)
        confidence = probabilities[prediction] * 100
        
        self.prediction_label.config(text=f'Prédiction: {prediction}', font=('Arial', 20, 'bold'))
        
        top3 = np.argsort(probabilities)[-3:][::-1]
        prob_text = f'Confiance: {confidence:.1f}% | Top 3: '
        for idx in top3:
            prob_text += f'{idx}({probabilities[idx]*100:.1f}%) '
        self.prob_label.config(text=prob_text)


if __name__ == "__main__":
    app_root = tk.Tk()
    DigitDrawer(app_root, loaded_model)
    app_root.mainloop()
