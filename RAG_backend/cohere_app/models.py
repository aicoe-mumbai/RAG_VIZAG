from django.db import models
import uuid
from django.contrib.auth.models import User

# models.py
class PromptHistory(models.Model):
    USECASE_CHOICES = [
        ('chat', 'chat'),
        ('qa',   'qa'),
    ]
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    session_id = models.UUIDField(default=uuid.uuid4, editable=False)
    usecase = models.CharField(
        max_length=10,
        choices=USECASE_CHOICES,
        default='chat',
        help_text='The type of interaction: chat or QA.'
    )
    prompt = models.TextField()
    response = models.TextField()
    comments = models.TextField(null=True, blank=True)
    thumbs_feedback = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user.username}, Usecase: {self.usecase}, Prompt: {self.prompt}"


class CurrentUsingCollection(models.Model):
    current_using_collection = models.CharField(max_length=255, unique=True, blank=True)
    def __str__(self):
        return self.current_using_collection

class UserInfo(models.Model):
    username   = models.CharField(max_length=255)

    ROLE_CHOICES = [
        ('user',  'User'),
        ('admin', 'Admin'),
    ]
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, default='user')

    USECASE_CHOICES = [
        ('chat', 'Chat'),
        ('qa',   'QA'),
    ]
    usecase = models.CharField(
        max_length=10,
        choices=USECASE_CHOICES,
        null=True,
        blank=True,
        help_text="Leave null for most users; set to 'qa' in login for your selected users."
    )

    def __str__(self):
        return f"UserInfo[{self.username}] role={self.role} usecase={self.usecase}"
