from django.views import View
from django.shortcuts import redirect
from django.views.generic import TemplateView
from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth import authenticate, login, logout

User = get_user_model()


class UserCreateView(TemplateView):
    template_name = "register.html"

    def get(self, request, *args, **kwargs):
        if request.user.is_authenticated:
            return redirect("data_presenter:data_presentation")
        return super(UserCreateView, self).get(request, *args, **kwargs)

    def post(self, request):
        email = request.POST.get("email")
        user_name = request.POST.get("user_name")
        User.objects.create_user(email=email, name=user_name)
        user = authenticate(email=email)
        if user is not None:
            login(request, user)
            return redirect("data_presenter:data_presentation")
        return redirect("register")


class LoginView(TemplateView):
    template_name = "login.html"

    def get(self, request, *args, **kwargs):
        next_page = request.GET.get("next")
        if request.user.is_authenticated:
            if next_page:
                return redirect(next_page)
            return redirect("data_presenter:data_presentation")
        return super(LoginView, self).get(request, *args, **kwargs)

    def post(self, request):
        next_page = request.GET.get("next", "data_presenter:data_presentation")
        email = request.POST.get("email")
        user = authenticate(email=email)
        if user:
            login(request, user)
        return redirect(next_page)


class LogoutView(LoginRequiredMixin, View):
    login_url = "login"

    def get(self, request):
        logout(request)
        return redirect("login")
