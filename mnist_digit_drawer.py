"""
Application Pygame pour dessiner des chiffres MNIST
"""

import pygame
import numpy as np
from PIL import Image
from tensorflow import keras

# Configuration
WIDTH, HEIGHT = 600, 700
CANVAS_SIZE = 400
CANVAS_POS = (100, 100)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (76, 175, 80)
RED = (244, 67, 54)

# TODO: Chargement du modèle
# model = keras.models.load_model('mnist_cnn_model.h5')


def predict_digit(image_array):
    """Prédit le chiffre"""
    image_array = image_array.astype('float32') / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)
    probabilities = model.predict(image_array, verbose=0)
    prediction = np.argmax(probabilities)
    return prediction, probabilities[0]


class DigitDrawerApp:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Dessinateur MNIST")
        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self.canvas.fill(WHITE)
        
        self.drawing = False
        self.last_pos = None
        self.prediction_text = "Dessinez un chiffre"
        self.prob_text = ""
        
        self.predict_button = pygame.Rect(100, 550, 180, 50)
        self.clear_button = pygame.Rect(320, 550, 180, 50)
    
    def draw_button(self, rect, text, color, hover=False):
        """Dessine un bouton"""
        button_color = tuple(min(255, c + 30) for c in color) if hover else color
        pygame.draw.rect(self.screen, button_color, rect, border_radius=10)
        pygame.draw.rect(self.screen, BLACK, rect, 2, border_radius=10)
        
        text_surface = self.font_medium.render(text, True, WHITE)
        text_rect = text_surface.get_rect(center=rect.center)
        self.screen.blit(text_surface, text_rect)
    
    def clear_canvas(self):
        """Efface le canvas"""
        self.canvas.fill(WHITE)
        self.prediction_text = "Dessinez un chiffre"
        self.prob_text = ""
    
    def predict(self):
        """Fait une prédiction"""
        # Convertir le canvas en array
        canvas_array = pygame.surfarray.array3d(self.canvas)
        canvas_array = canvas_array[:, :, 0].T
        
        # Redimensionner à 28x28
        img = Image.fromarray(canvas_array.astype('uint8'))
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Inverser les couleurs
        img_array = 255 - np.array(img)
        
        # Prédiction
        prediction, probabilities = predict_digit(img_array)
        confidence = probabilities[prediction] * 100
        
        self.prediction_text = f"Prédiction: {prediction}"
        
        # Top 3
        top3 = np.argsort(probabilities)[-3:][::-1]
        self.prob_text = f"Confiance: {confidence:.1f}% | Top 3: "
        for idx in top3:
            self.prob_text += f"{idx}({probabilities[idx]*100:.1f}%) "
    
    def handle_events(self):
        """Gère les événements"""
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                rel_pos = (mouse_pos[0] - CANVAS_POS[0], mouse_pos[1] - CANVAS_POS[1])
                if 0 <= rel_pos[0] < CANVAS_SIZE and 0 <= rel_pos[1] < CANVAS_SIZE:
                    self.drawing = True
                    self.last_pos = rel_pos
                
                if self.predict_button.collidepoint(mouse_pos):
                    self.predict()
                elif self.clear_button.collidepoint(mouse_pos):
                    self.clear_canvas()
            
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.drawing = False
                self.last_pos = None
            
            elif event.type == pygame.MOUSEMOTION and self.drawing:
                rel_pos = (mouse_pos[0] - CANVAS_POS[0], mouse_pos[1] - CANVAS_POS[1])
                if 0 <= rel_pos[0] < CANVAS_SIZE and 0 <= rel_pos[1] < CANVAS_SIZE:
                    if self.last_pos:
                        pygame.draw.line(self.canvas, BLACK, self.last_pos, rel_pos, 20)
                    pygame.draw.circle(self.canvas, BLACK, rel_pos, 10)
                    self.last_pos = rel_pos
        
        return True
    
    def draw(self):
        """Dessine l'interface"""
        self.screen.fill(WHITE)
        mouse_pos = pygame.mouse.get_pos()
        
        # Titre
        title = self.font_large.render("Dessinateur MNIST", True, BLACK)
        self.screen.blit(title, title.get_rect(center=(WIDTH // 2, 40)))
        
        # Canvas
        pygame.draw.rect(self.screen, BLACK, (CANVAS_POS[0]-3, CANVAS_POS[1]-3, CANVAS_SIZE+6, CANVAS_SIZE+6), 3)
        self.screen.blit(self.canvas, CANVAS_POS)
        
        # Boutons
        self.draw_button(self.predict_button, "Prédire", GREEN, self.predict_button.collidepoint(mouse_pos))
        self.draw_button(self.clear_button, "Effacer", RED, self.clear_button.collidepoint(mouse_pos))
        
        # Résultats
        pred = self.font_large.render(self.prediction_text, True, BLACK)
        self.screen.blit(pred, pred.get_rect(center=(WIDTH // 2, 630)))
        
        if self.prob_text:
            prob = self.font_small.render(self.prob_text, True, BLACK)
            self.screen.blit(prob, prob.get_rect(center=(WIDTH // 2, 670)))
        
        pygame.display.flip()
    
    def run(self):
        """Boucle principale"""
        running = True
        while running:
            running = self.handle_events()
            self.draw()
            self.clock.tick(60)
        pygame.quit()


if __name__ == "__main__":
    app = DigitDrawerApp()
    app.run()
