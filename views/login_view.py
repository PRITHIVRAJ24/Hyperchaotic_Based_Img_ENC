import flet as ft
from views.encrypt_view import encryption_view
from views.decrypt_view import decryption_view
def login_view(page: ft.Page):
    page.clean()
    page.title = "PixelFlex - Login"
    
    def authenticate(e):
        if not user_name.value or not pwd.value:
            user_name.error_text = "Please enter your username"
            pwd.error_text = "Please enter your password"
        elif user_name.value == "admin" and pwd.value == "admin":
            if dd.value == "Encrypt":
                encryption_view(page)
            elif dd.value == "Decrypt":
                decryption_view(page)
            pass
        else:
            page.banner = ft.Banner(
                bgcolor=ft.colors.WHITE,
                leading=ft.Icon(ft.icons.WARNING_AMBER_ROUNDED, color=ft.colors.RED, size=40),
                content=ft.Text("Invalid Username or Password!", color=ft.colors.GREEN),
                actions=[ft.TextButton("Retry", on_click=lambda e: setattr(page.banner, 'open', False))]
            )
            page.banner.open = True
        page.update()
    
    user_name = ft.TextField(label="Username")
    pwd = ft.TextField(label="Password", password=True)
    dd = ft.Dropdown(
        width=100,
        options=[ft.dropdown.Option("Encrypt"), ft.dropdown.Option("Decrypt")],
    )
    
    page.add(user_name, pwd, dd, ft.ElevatedButton("Login", on_click=authenticate))