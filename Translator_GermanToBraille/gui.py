

import braille_viaUnicode as bu
import tkinter as tk


## initiate window
window = tk.Tk()
window.title("Translator: German - Braille")


##### DEFINE FRAMES ########################################################################

## define frame for text input
frame_txt_input = tk.Frame(master=window, height=100, bg="white")
frame_txt_input.pack(fill=tk.Y, side=tk.LEFT, expand=True)

## define frame for text output
frame_txt_output = tk.Frame(master=window, height=100, bg="white")
frame_txt_output.pack(fill=tk.Y, side=tk.LEFT, expand=True)

## define frame for buttons
frame_btn = tk.Frame(master=window, height=10, bg="gray")
frame_btn.pack(fill=tk.Y, side=tk.LEFT, expand=True)

## define rows and columns for grid
window.rowconfigure([0, 1], minsize=50, weight=1)
window.columnconfigure([0, 1], minsize=50, weight=1)


##### Define input / output text fields ######################################################

input_text = tk.Text(master=frame_txt_input)
input_text.grid(row=0, column=0)

output_text = tk.Text(master=frame_txt_input)
output_text.grid(row=1, column=0)


##### Define buttons #########################################################################

## button for translating German to Braille symbols
def handleClick_toBraille():
    value = input_text.get("1.0", tk.END).replace("\n", " ")
    value2 = bu.translateToBraille(value)
    output_text.insert("1.0", "\n"+value2)
btn_toBraille = tk.Button(master=frame_btn,
                          text="Braille",
                          width=25,
                          fg="blue",
                          command=handleClick_toBraille)
btn_toBraille.grid(row=0, column=1)

## button for deleting text in text field
def handleClick_deleteText():
    input_text.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)
btn_deleteText = tk.Button(master=frame_btn,
                           text="Delete Text",
                           width=25,
                           fg="red",
                           command=handleClick_deleteText)
btn_deleteText.grid(row=1, column=1)


def main():
    window.mainloop()


if __name__ == "__main__":
    main()
