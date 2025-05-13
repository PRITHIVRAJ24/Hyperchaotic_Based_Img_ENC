import flet as ft
import base64
from src.encrypt import encrypt_image

def encryption_view(page: ft.Page):
    page.clean()
    page.title = "Encryption"

    file_path_en = ft.Text(value="No file selected", size=16, color=ft.colors.BLACK)
    file_preview_label = ft.Text("Original Image", size=18, weight=ft.FontWeight.BOLD, visible=False)
    processed_preview_label = ft.Text("Encrypted Image", size=18, weight=ft.FontWeight.BOLD, visible=False)

    file_preview = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False)  # Left
    processed_preview = ft.Image(width=300, height=300, fit=ft.ImageFit.CONTAIN, visible=False)  # Right

    # aes_key = ft.TextField(label="Enter your Key", password=True, filled=True, border_radius=10)
    # con_aes_key = ft.TextField(label="Re-enter your Key", password=True, filled=True, border_radius=10)

    selected_image_path = None  

    get_file_dialog_en = ft.FilePicker(on_result=lambda e: select_image(e))
    page.overlay.append(get_file_dialog_en)

    def select_image(e):

        nonlocal selected_image_path
        if e.files:
            selected_image_path = e.files[0].path
            file_path_en.value = selected_image_path
            display_image(selected_image_path, file_preview)  
            file_preview.visible = True
            file_preview_label.visible = True
        else:
            selected_image_path = None
            file_path_en.value = "No file selected"

        file_path_en.update()
        file_preview.update()
        file_preview_label.update()

    def encrypt_and_display(e):

        if not selected_image_path:
            file_path_en.value = "Please select an image first!"
            file_path_en.update()
            return

        encrypted_data = encrypt_image(selected_image_path)
  
        display_image(encrypted_data, processed_preview)
        processed_preview.visible = True 
        processed_preview_label.visible = True
        processed_preview.update()
        processed_preview_label.update()

    def display_image(image_path, preview):
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                preview.src_base64 = encoded_string
        except Exception as ex:
            preview.src_base64 = None
            file_path_en.value = f"Error loading image: {ex}"
        
        preview.update()

    def go_back(e):
       
        from views.login_view import login_view
        login_view(page)

   
    layout = ft.Column(
        controls=[
            ft.Text("Encryption Page", size=24, weight=ft.FontWeight.BOLD, color=ft.colors.BLUE_700),
            ft.ElevatedButton("Select Image", icon=ft.icons.IMAGE, on_click=lambda _: get_file_dialog_en.pick_files(allow_multiple=False)),
            file_path_en,
            ft.Row(
                controls=[
                    ft.ElevatedButton("Confirm", on_click=encrypt_and_display, bgcolor=ft.colors.GREEN_400, color=ft.colors.WHITE),
                    ft.ElevatedButton("Back", on_click=go_back, bgcolor=ft.colors.RED_400, color=ft.colors.WHITE)
                ],
                alignment=ft.MainAxisAlignment.SPACE_EVENLY
            ),
            ft.Row(
                controls=[
                    ft.Column([file_preview_label, file_preview], alignment=ft.MainAxisAlignment.CENTER),
                    ft.Column([processed_preview_label, processed_preview], alignment=ft.MainAxisAlignment.CENTER),
                ],
                alignment=ft.MainAxisAlignment.SPACE_EVENLY,
            ),
        ],
        spacing=20
    )

    page.add(ft.Container(content=layout, padding=20))
