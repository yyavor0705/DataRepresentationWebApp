from django.contrib.auth.backends import ModelBackend

from .models import User


class CustomerBackend(ModelBackend):

    def authenticate(self, request, **kwargs):
        user_email = kwargs.get('email')
        try:
            user = User.objects.get(email=user_email)
            return user
        except User.DoesNotExist:
            pass
