import tkinter as tk
import requests

def submit_input(event=None):
    user_input = entry.get().strip()
    if not user_input:
        return
    try:
        response = requests.post("http://localhost:9000/push_input", json={"message": user_input})
        result_label.config(text="Server response: " + str(response.json()))
    except Exception as e:
        result_label.config(text="Error: " + str(e))
    entry.delete(0, tk.END)

root = tk.Tk()
root.title("用户输入界面")

# 创建输入框，并绑定回车键事件自动提交
entry = tk.Entry(root, font=("SimHei", 16), width=40)
entry.pack(padx=20, pady=10)
entry.bind("<Return>", submit_input)

# 用于显示服务器返回结果的标签
result_label = tk.Label(root, text="", font=("SimHei", 14))
result_label.pack(padx=20, pady=10)

root.mainloop()
