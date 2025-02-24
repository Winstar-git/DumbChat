#James Starwin T. Canoy
#BSCPE 1-2
#Feb 24, 2025

from transformers import AutoModelForCausalLM, AutoTokenizer
import tkinter as tk
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "microsoft/DialoGPT-medium"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)


def chatbot_response(user_input, chat_history_ids=None):
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    if chat_history_ids is None:
        chat_history_ids = new_user_input_ids
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    attention_mask = torch.ones(chat_history_ids.shape, dtype=torch.long)

    text_output = model.generate(
    chat_history_ids,
    do_sample=True,
    max_length=100,  
    num_return_sequences=1,
    no_repeat_ngram_size=3, 
    repetition_penalty=1.1,
    temperature=0.3,
    top_k=40,
    top_p=0.85,
    pad_token_id=tokenizer.eos_token_id,
    attention_mask=attention_mask 
)


    bot_response = tokenizer.decode(text_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)
    chat_history_ids = chat_history_ids[:, -1024:]
    return bot_response, chat_history_ids


root = tk.Tk()
root.title("DumbGPT Chatbot")

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

window_width = 600
window_height = 450

position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)

root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")


chat_history = None
chat_window = tk.Text(root, height=20, width=50, state=tk.DISABLED, bg="white", font=("Arial", 12))
chat_window.grid(row=0, column=0, padx=10, pady=10)

user_input_entry = tk.Entry(root, width=40, font=("Arial", 12))
user_input_entry.grid(row=1, column=0, padx=10, pady=10)


def intro():
    chat_window.config(state=tk.NORMAL)
    chat_window.insert(tk.END, "DumbBot: Hello! I'm DumbBot.\n")
    chat_window.yview(tk.END)
    chat_window.config(state=tk.DISABLED)
intro()

def send_message():
    global chat_history
    user_message = user_input_entry.get()
    if user_message.strip():
        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "You: " + user_message + "\n")
        chat_window.yview(tk.END)
        chat_window.config(state=tk.DISABLED)
        bot_response, chat_history = chatbot_response(user_message, chat_history)

        chat_window.config(state=tk.NORMAL)
        chat_window.insert(tk.END, "DumbBot: " + bot_response + "\n")
        chat_window.yview(tk.END)
        chat_window.config(state=tk.DISABLED)
    user_input_entry.delete(0, tk.END)

send_button = tk.Button(root, text="Send", width=10, font=("Arial", 12), command=send_message)
send_button.grid(row=1, column=1, padx=10, pady=10)

root.mainloop()
