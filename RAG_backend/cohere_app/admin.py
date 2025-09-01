from django.contrib import admin
from .models import PromptHistory, CurrentUsingCollection


class DataAdmin(admin.ModelAdmin):
    # Specify the fields to display as columns
    list_display = ('user', 'session_id', 'prompt', 'response', 'comments', 'created_at')
    

# Register the model and the admin class
admin.site.register(PromptHistory, DataAdmin)
admin.site.register(CurrentUsingCollection)