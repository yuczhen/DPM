from django.shortcuts import redirect


class PasswordProtectMiddleware:
    """
    簡易密碼保護中介層。
    未驗證的請求一律導向 /login/，靜態檔案由 WhiteNoise 在更前面處理所以不受影響。
    """

    EXEMPT_PATHS = ("/login/",)

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if not request.session.get("authenticated"):
            if request.path not in self.EXEMPT_PATHS:
                return redirect("prediction:login")
        return self.get_response(request)
