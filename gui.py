import tkinter as tk
import tkinter.messagebox
import customtkinter
from tkvideo import tkvideo
import vlc

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class Screen(tk.Frame):
    '''
    Screen widget: Embedded video player from local or youtube
    '''

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, bg='black')
        self.parent = parent
        # Creating VLC player
        self.instance = vlc.Instance()
        self.player = self.instance.media_player_new()

    def GetHandle(self):
        # Getting frame ID
        return self.winfo_id()

    def play(self, _source):
        # Function to start player from given source
        Media = self.instance.media_new(_source)
        Media.get_mrl()
        self.player.set_media(Media)

        
        self.player.set_hwnd(self.winfo_id())
        self.player.play()


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        # configure window
        self.title("CREAR ESTO ME HA HECHO ENVEJECER 54 AÑOS")
        self.geometry(f"{1200}x{720}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure((2, 3), weight=0)
        self.grid_rowconfigure((0, 1, 2), weight=1)

        # Create videoFrame holder
        self.videoHolder = tk.Frame(self)
        self.videoHolder.grid(row=0, column=1, padx=20, pady=(10, 10),sticky="nsew",rowspan=4)
        # Init vlc player
        self.player = Screen(self.videoHolder)
        self.player.place(relx=0.0005, rely=0, relwidth=0.999, relheight=1)
        self.player.play('dos_personas_hablando_por_turnos.mp4')



        self.save_button = customtkinter.CTkButton(master=self, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Save",command=self.saveSample)
        self.save_button.grid(row=7, column=1, padx=(20, 20), pady=(20, 20))

        self.accept_button = customtkinter.CTkButton(master=self, fg_color="#adedbe", border_width=2, text_color=("gray10", "#DCE4EE"), text="Accept",command=self.saveSample)
        self.accept_button.grid(row=0, column=2, padx=(20, 20))

        self.incorrect_button = customtkinter.CTkButton(master=self, fg_color="#edadad", border_width=2, text_color=("gray10", "#DCE4EE"), text="Incorrect",command=self.saveSample)
        self.incorrect_button.grid(row=1, column=2, padx=(20, 20))

        # create textbox
        self.textbox = customtkinter.CTkTextbox(self, width=250)
        self.textbox.grid(row=4, column=1, padx=(20, 0), pady=(20, 0), sticky="nsew")

        # set default values
        #self.appearance_mode_optionemenu.set("Dark")
        self.textbox.insert("0.0", "---Video transcription---")

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def saveSample(self):
        self.textbox.delete("0.0","end")
        self.textbox.insert("0.0", "TEST")
        self.player.play('dos_personas_hablando_por_turnos.mp4')



if __name__ == "__main__":
    app = App()
    app.mainloop()
