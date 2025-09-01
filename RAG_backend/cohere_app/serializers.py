# serializers.py
from rest_framework import serializers
from .models import PromptHistory,CurrentUsingCollection

class PromptHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = PromptHistory
        fields = ['id', 'session_id', 'prompt', 'response', 'created_at'] 


class CurrentUsingCollectionSerializer(serializers.ModelSerializer):
    class Meta:
        model = CurrentUsingCollection
        fields = ['current_using_collection']
