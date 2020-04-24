from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class DataModelPickle(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    pickle = models.BinaryField()
