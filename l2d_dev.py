import pygame
import live2d.v3 as live2d
from pygame.locals import *
from OpenGL.GL import *
import ctypes
import time
import threading
import requests
import sys
from langserve import RemoteRunnable
from langchain.schema import HumanMessage

rag_agent = RemoteRunnable("http://localhost:8000/chat")

# ----------------- 修改后的 get_text -----------------
def get_text():
    """
    从消息服务器拉取下一条消息:
    - 如果队列中有消息, 返回其字符串内容
    - 如果队列为空, 返回 None
    """
    try:
        resp = requests.get("http://localhost:9000/pop_message")
        data = resp.json()
        msg = data.get("message")
        return msg  # 可能是字符串或None
    except Exception as e:
        print("Failed to get message:", e)
        return None

# ----------------- 定义一个请求函数，用线程调用 -----------------
def request_text():
    global text_full, text_display
    new_text = get_text()
    print("Received new text:", new_text)
    if new_text:
        text_full = new_text
        text_display = new_text  # 直接显示完整文本

# Initialize Pygame and OpenGL
pygame.init()
display = (800, 600)

# Create OpenGL window
screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
glClearColor(0.0, 1.0, 0.0, 1.0)
pygame.display.set_caption("Live2D Transparent Window")

# Enable window transparency (Windows only)
window_id = pygame.display.get_wm_info()["window"]
ctypes.windll.user32.SetWindowPos(window_id, -1, 0, 0, 0, 0, 0x0001)

# Initialize Live2D
live2d.init()
live2d.glewInit()

# Load Live2D model
model = live2d.LAppModel()
model.LoadModelJson("./data/l2d/model3.json")
model.Resize(800, 600)

# Initialize text box
pygame.font.init()
font = pygame.font.Font("C:/Windows/Fonts/simhei.ttf", 30)  # Windows

# Set text box at the bottom
text_box = pygame.Rect(50, 500, 700, 80)  # Positioned lower
text_full = ""    # Full text received
text_display = "" # Directly display the full text
# 去除逐字显示变量
# char_index = 0
text_end_time = None  # Timer for keeping text before clearing
request_interval = 1  # 每隔1秒发送一次请求
last_request_time = time.time()

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

    current_time = time.time()
    # 每隔1秒发送请求获取新文本（仅当当前没有显示文本时）
    if not text_full and current_time - last_request_time >= request_interval:
        last_request_time = current_time
        # 使用线程异步调用 get_text
        threading.Thread(target=request_text, daemon=True).start()

    # 如果文本已显示超过10秒，则清空（可以按需调整）
    if text_full and text_end_time is None:
        text_end_time = current_time  # 开始计时
    if text_end_time and current_time - text_end_time > 10:
        text_full = ""
        text_display = ""
        text_end_time = None

    # Render Live2D model
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    live2d.clearBuffer(0, 1, 0, 0)
    angle_x = max(-30, min(30, angle_x))
    model.Update()
    model.SetParameterValue("ParamAngleX", angle_x, 1.0)
    model.Draw()

    # Render text box if there's text
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

        # Render text onto overlay
        y_offset = text_box.y + 10
        for line in lines:
            text_surface = font.render(line, True, (255, 255, 255))
            overlay.blit(text_surface, (text_box.x + 10, y_offset))
            y_offset += 30

        # Flip overlay for OpenGL
        overlay = pygame.transform.flip(overlay, False, True)
        texture_data = pygame.image.tostring(overlay, "RGBA", True)

        glBindTexture(GL_TEXTURE_2D, gui_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, display[0], display[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

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
