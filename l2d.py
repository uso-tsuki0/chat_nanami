import pygame
import live2d.v3 as live2d
from pygame.locals import *
from OpenGL.GL import *
import ctypes
import time

# Simulate OpenAI API streaming response
import random

def get_text():
    """ Simulate streaming text from OpenAI API. """
    messages = [
        "Hello! Welcome to the Live2D system.",
        "Pygame supports multiple rendering methods.",
        "Live2D can create animated characters!",
        "You can get text from an external source.",
        "The text box is semi-transparent and supports word wrap.",
        None  # Simulate no message sometimes
    ]
    return random.choice(messages)

# Initialize Pygame and OpenGL
pygame.init()
display = (800, 600)

# Create OpenGL window
screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
pygame.display.set_caption("Live2D Transparent Window")

# Enable window transparency (Windows only)
window_id = pygame.display.get_wm_info()["window"]
ctypes.windll.user32.SetWindowLongW(window_id, -20, ctypes.windll.user32.GetWindowLongW(window_id, -20) | 0x00080000)
ctypes.windll.user32.SetLayeredWindowAttributes(window_id, 0, 0, 1)

# Initialize Live2D
live2d.init()
live2d.glewInit()

# Load Live2D model
model = live2d.LAppModel()
model.LoadModelJson("./data/l2d/model3.json")
model.Resize(800, 600)

# Enable OpenGL transparency
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
glClearColor(0.0, 0.0, 0.0, 0.0)

# Initialize text box
pygame.font.init()
font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 30)  # Windows
# font = pygame.font.Font("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 30)  # Linux
# font = pygame.font.SysFont("simhei", 30)  # Mac

# Set text box at the bottom
text_box = pygame.Rect(50, 500, 700, 80)  # Positioned lower
text_full = ""   # Full text received
text_display = ""  # Gradually displayed text
char_index = 0  # Current index for revealing text
text_end_time = None  # Timer for keeping text before clearing
char_delay = 0.05  # Delay between each character
last_char_time = time.time()

# OpenGL texture for GUI rendering
gui_texture = glGenTextures(1)

# Main loop
running = True
angle_x = 0

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_a:  # Press 'A' to change expression
                model.SetRandomExpression()

    # Fetch new text when current text is cleared
    if text_full == "" and (text_end_time is None or time.time() - text_end_time > 10):
        new_text = get_text()
        if new_text:
            text_full = new_text
            text_display = ""
            char_index = 0
            text_end_time = None  # Reset timer

    # Gradually reveal characters
    if char_index < len(text_full) and time.time() - last_char_time > char_delay:
        text_display = text_full[:char_index + 1]  # Show one more character
        char_index += 1
        last_char_time = time.time()

    # If text is fully displayed, start the countdown to clear
    if char_index == len(text_full) and text_end_time is None:
        text_end_time = time.time()  # Start 10s timer

    # Clear text after 10 seconds
    if text_end_time and time.time() - text_end_time > 10:
        text_full = ""
        text_display = ""

    # Render Live2D model
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    live2d.clearBuffer()
    angle_x = max(-30, min(30, angle_x))
    model.Update()
    model.SetParameterValue("ParamAngleX", angle_x, 1.0)
    model.Draw()

    # Render text box
    if text_display:
        overlay = pygame.Surface(display, pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 0))  # Transparent background

        # Draw semi-transparent text box
        pygame.draw.rect(overlay, (50, 50, 50, 180), text_box, border_radius=10)

        # Word wrap
        words = text_display.split(" ")
        lines = []
        current_line = ""

        for word in words:
            if font.size(current_line + word)[0] < text_box.width - 20:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)

        # Render text
        y_offset = text_box.y + 10
        for line in lines:
            text_surface = font.render(line, True, (255, 255, 255))
            overlay.blit(text_surface, (text_box.x + 10, y_offset))
            y_offset += 30

        # Flip the overlay to match OpenGL texture orientation
        overlay = pygame.transform.flip(overlay, False, True)

        # Update OpenGL texture
        texture_data = pygame.image.tostring(overlay, "RGBA", True)

        glBindTexture(GL_TEXTURE_2D, gui_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 800, 600, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Render texture in OpenGL
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, gui_texture)

        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)  # Bottom left
        glTexCoord2f(1, 1); glVertex2f( 1, -1)  # Bottom right
        glTexCoord2f(1, 0); glVertex2f( 1,  1)  # Top right
        glTexCoord2f(0, 0); glVertex2f(-1,  1)  # Top left
        glEnd()

        glDisable(GL_TEXTURE_2D)

    # Swap OpenGL buffers
    pygame.display.flip()

# Cleanup
live2d.dispose()
pygame.quit()
