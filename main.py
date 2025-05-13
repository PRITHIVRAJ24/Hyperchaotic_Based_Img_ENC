import flet as ft
from flet import icons, FilePicker, FilePickerResultEvent
from views.login_view import login_view
from views.encrypt_view import encryption_view
from views.decrypt_view import decryption_view

def main(page: ft.Page):
    # encryption_view(page)
    login_view(page)
    #decryption_view(page)

if __name__ == "__main__":
    ft.app(target=main)