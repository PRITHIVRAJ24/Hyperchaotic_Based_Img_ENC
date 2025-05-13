import flet as ft
from flet import icons, FilePicker, FilePickerResultEvent
from src.decrypt import decrypt_image

def decryption_view(page: ft.Page):
    page.clean()
    page.title = "Decryption"

    file_path_de = ft.Text()
    
    def decrypt(e):
        print(f"Decrypting: {file_path_de.value}")
        
        decrypted_image.src = decrypt_image(file_path_de.value) 
        decrypted_image.update()
        if decrypted_image.src:  
            decrypted_label.visible = True
            decrypted_label.update()

    def get_file_result_de(e: FilePickerResultEvent):
        file_path_de.value = e.files[0].path if e.files else "Cancelled!"
        image_preview.src = file_path_de.value  
        image_preview.update()
        if image_preview.src:  
            encrypted_label.visible = True
            encrypted_label.update()
    
    def go_back(e):
        from views.login_view import login_view
        login_view(page)
    
    get_file_dialog_de = FilePicker(on_result=get_file_result_de)
    page.overlay.append(get_file_dialog_de)

    # password_visible = False
    
    # def toggle_password_visibility(e):
    #     nonlocal password_visible
    #     password_visible = not password_visible
    #     de_aes_key.password = not password_visible  
    #     de_aes_key.update()
        
    #     view_password_button.icon = icons.VISIBILITY if password_visible else icons.VISIBILITY_OFF
    #     view_password_button.update()

    
    # de_aes_key = ft.TextField(label="Enter your passkey", password=True)

    # view_password_button = ft.IconButton(icon=icons.VISIBILITY_OFF, on_click=toggle_password_visibility)

    
    encrypted_label = ft.Text("Encrypted Image", weight=ft.FontWeight.BOLD, visible=False)
    decrypted_label = ft.Text("Decrypted Image", weight=ft.FontWeight.BOLD, visible=False)
    

    image_preview = ft.Image(width=200, height=200, fit=ft.ImageFit.CONTAIN)
    decrypted_image = ft.Image(width=200, height=200, fit=ft.ImageFit.CONTAIN)
    
   
    row_layout = ft.Row(
        controls=[
            ft.Column(controls=[
                encrypted_label,
                image_preview
            ], spacing=10),  
            ft.Column(controls=[
                decrypted_label,
                decrypted_image
            ], spacing=10) 
        ],
        alignment=ft.MainAxisAlignment.CENTER 
    )

    page.add(ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Decryption Page", size=24, weight=ft.FontWeight.BOLD),
                ft.ElevatedButton("Select Image", icon=icons.IMAGE, on_click=lambda _: get_file_dialog_de.pick_files(allow_multiple=False)),
                file_path_de,  
                row_layout,  
                ft.ElevatedButton("Decrypt", on_click=decrypt),
                ft.ElevatedButton("Back", on_click=go_back)  
            ],
            alignment=ft.MainAxisAlignment.START,
            spacing=20
        ),
        padding=20,  
        bgcolor=ft.colors.WHITE 
    ))
