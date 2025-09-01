from django.urls import path
from .views import *

urlpatterns = [
    path('qa/admin/', qa_admin, name='qa-admin'),
    path('cohere/generate/', cohere_generate, name='cohere_generate'),
    path('history/', get_prompt_history, name='get_saved_prompts'),
    path('history/<str:session_id>/', get_session_history, name='get_session_history'),  
    path('login/', login_user, name='login'),
    path('logout/', logout_user, name='logout'),
    path('token/refresh/', refresh_access_token, name='token-refresh'),
    path('upload_file_from_the_user/',file_upload_view),
    path('save-comment/<int:pk>/',save_comment, name='save-comment'),
    path('mark_satisfied_or_unsatisfied/<int:pk>/', mark_satisfied_or_unsatisfied, name='mark_unsatisfied'),
    path('documents/', get_documents),
    path('get-folder/',get_folder_name),
    path('serve-file/<path:filename>/<int:page_number>/', serve_file, name='serve_pdf'),
    path('collections/', get_collection_name, name='get_collections'),
    path('collections/<str:collection_name>/files/', collection_files, name='collection-files'),
    path('collections/<str:collection_name>/delete/', delete_collection, name='delete-collection'),
    path('collections/file-delete/<path:source>/<str:collection_name>/', delete_file, name='delete_file'),
    path('collections/create_collection/', create_collection, name='create_collection'),
    path('collections/progress/', get_progress, name='get_progress'),
    path("milvus-data/<str:collection_name>/", get_milvus_data, name="milvus-data"),
    path('current-using-collection/', get_current_using_collection, name='get-current-using-collection'),
    path("update-current-collection/", update_current_collection, name="update-current-collection"),
    path('restart-server/', restart_server, name='restart_server')
]