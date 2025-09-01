from decouple import config
import os, json, mimetypes
from datetime import datetime, timedelta
import glob
import uuid
import subprocess
import urllib.parse
import threading
from django.contrib.auth import get_user_model
from pymilvus import connections, Collection, MilvusClient
from django.http import JsonResponse, StreamingHttpResponse, FileResponse, Http404
from django.contrib.auth import authenticate
from django.utils import timezone
from django.db import connection
from django.db.models import Min
from django.views.decorators.csrf import csrf_exempt
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response
from rest_framework_simplejwt.tokens import RefreshToken
from .models import PromptHistory, CurrentUsingCollection
from .models import UserInfo
from .serializers import PromptHistorySerializer, CurrentUsingCollectionSerializer
from .api import process_query, get_all_files_from_milvus, chat_with_uploaded_document
from .Chunking_UI import db_utility, file_process_new
from .Chunking_UI.enable_logging import logger
from urllib.parse import unquote
from dotenv import load_dotenv
from .ldap import auth_main
import tempfile
import shutil
from django.contrib.auth.models import User
load_dotenv()

Milvus_url = os.getenv("MILVUS_URL")
client = MilvusClient(uri= Milvus_url, token="root:Milvus")
progress_data = {"message": "Starting upload..."}

PDF_DIRECTORY = config('PDF_DIRECTORY')
host = os.getenv("HOST")
port = os.getenv("PORT")


@api_view(['POST'])
@permission_classes([AllowAny])
def login_user(request):
    username = request.data.get("username")
    password = request.data.get("password")

    # Hard-coded lists of special users
    QA_USERS    = ["456"]  # Users granted QA access
    ADMIN_USERS = ["456"]  # Users granted Admin access

    try:
        # Authenticate via LDAP
        user = auth_main(username=username, password=password)
        if not user:
            return Response({"error": "Invalid credentials"},
                            status=status.HTTP_401_UNAUTHORIZED)

        # Ensure Django User exists
        UserModel = get_user_model()
        user_instance, _ = UserModel.objects.get_or_create(username=username)

        # Issue JWT tokens
        refresh = RefreshToken.for_user(user_instance)

        # Create or update the UserInfo row;
        # don't set usecase here so it remains NULL by default (chat)
        session, _ = UserInfo.objects.get_or_create(
            username=username,
            defaults={'role': 'user'}
        )

        # 1) Default all users to role='user', usecase=None (chat)
        session.role    = 'user'
        session.usecase = None

        # 2) Grant QA to QA_USERS
        if username in QA_USERS:
            session.usecase = 'qa'

        # 3) Grant Admin to ADMIN_USERS
        if username in ADMIN_USERS:
            session.role = 'admin'

        session.save()

        # Return tokens and user context so front-end can store them
        return Response({
            'refresh':  str(refresh),
            'access':   str(refresh.access_token),
            'username': user_instance.username,
            'usecase':  session.usecase,  # either 'qa' or null
            'role':     session.role,     # either 'admin' or 'user'
        })

    except Exception as e:
        print(f'Login error: {e}')  # consider proper logging
        return Response({"error": "Authentication failed"},
                        status=status.HTTP_401_UNAUTHORIZED)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def logout_user(request):
    try:
        refresh_token = request.data.get("refresh")
        username = request.user.username
        desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", username)       
        if os.path.exists(desktop_path) and os.path.isdir(desktop_path):
            shutil.rmtree(desktop_path) 
            print(f"Folder '{username}' deleted from Desktop.")
        else:
            print(f"No folder named '{username}' found on Desktop.")
        token = RefreshToken(refresh_token)
        token.blacklist()
        return Response({"message": "Logout successful"}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"error": f"Invalid token or logout failed: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)


@api_view(['POST'])
def refresh_access_token(request):
    """
    Refresh the access token using the refresh token.
    """
    refresh_token = request.data.get('refresh')
    try:
        refresh = RefreshToken(refresh_token)
        new_access_token = str(refresh.access_token)
        return Response({"access": new_access_token}, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({"detail": f"Error refreshing token: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def cohere_generate(request):
    try:
        data        = request.data
        prompt      = data.get('prompt', '')
        session_id  = data.get('session_id') or str(uuid.uuid4())
        file_names  = data.get('file_names', [])
        jwt_token   = data.get('jwt_token', '')
        mode        = data.get('mode', 'chat')      # 'chat' or 'qa'
        user        = request.user
        user_upload = data.get('useruploadfile', '')

        # ─── Authorization check ────────────────────────────────────────────
        # Only users with UserInfo.usecase == 'qa' may call with mode='qa'
        try:
            user_info = UserInfo.objects.get(username=user.username)
        except UserInfo.DoesNotExist:
            return Response(
                {'error': 'User info not found'},
                status=status.HTTP_403_FORBIDDEN
            )

        if mode == 'qa' and user_info.usecase != 'qa':
            return Response(
                {'error': 'Not authorized for QA'},
                status=status.HTTP_403_FORBIDDEN
            )
        # ─────────────────────────────────────────────────────────────────────

        # flag only QA entries, leave chat entries unflagged
        usecase_flag = 'qa' if mode == 'qa' else None

        prompt_history_entry = PromptHistory.objects.create(
            user       = user,
            session_id = session_id,
            prompt     = prompt,
            response   = "",
            usecase    = usecase_flag
        )

        serializer = PromptHistorySerializer(prompt_history_entry)
        history_id = serializer.data['id']

        def response_stream():
            collected = []
            if user_upload:
                for partial in chat_with_uploaded_document(user.username, prompt):
                    collected.append(partial)
                    yield partial
            else:
                for partial in process_query(prompt, mode, file_names, jwt_token):
                    collected.append(partial)
                    yield partial

            combined = "\n".join(collected)
            yield "      "  # padding

            # save final response asynchronously
            def save_to_db():
                prompt_history_entry.response = combined
                prompt_history_entry.save()
            threading.Thread(target=save_to_db).start()

        response = StreamingHttpResponse(
            response_stream(),
            content_type='text/plain',
            status=status.HTTP_200_OK
        )
        response['X-History-ID'] = str(history_id)
        response['X-Session-ID'] = session_id
        response['Access-Control-Allow-Origin'] = '*'
        response['Access-Control-Expose-Headers'] = 'X-History-ID, X-Session-ID'

        return response

    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
@api_view(['GET', 'POST'])
@permission_classes([IsAuthenticated])
def qa_admin(request):
    """
    Admin‐only endpoint. Only users whose UserInfo.role == 'admin'
    will be allowed; everyone else gets a 403.
    """
    try:
        print("hello from qa_admin",request.user.username)
        user_info = UserInfo.objects.get(username=request.user.username)
        print("Hello from qa_admin2",user_info)
    except UserInfo.DoesNotExist:
        return Response({'error': 'UserInfo not found'}, status=status.HTTP_403_FORBIDDEN)

    if user_info.role != 'admin':
        return Response({'error': 'Not authorized for admin'}, status=status.HTTP_403_FORBIDDEN)

    # --- At this point you know it’s an admin ---
    # Put your real admin logic here. For now:
    return Response({
        'message': 'Welcome to QA Admin!',
        'username': request.user.username
    }, status=status.HTTP_200_OK)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def file_upload_view(request):
    uploaded_file = request.FILES.get('file')
    user_name = request.user.username
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:   
        temp_file.write(uploaded_file.read())           
        temp_file_path = temp_file.name
    faiss_response = file_process_new.create_faiss_index(temp_file_path,user_name)
    return JsonResponse({"message": "File uploaded successfully","temp_file_path": temp_file_path,"faiss_response": faiss_response})
 
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_prompt_history(request):
    
    today = timezone.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(weeks=1)
    last_month = today - timedelta(weeks=4)
    def get_first_prompts(queryset):
        first_prompts = queryset.values('session_id').annotate(first_prompt_id=Min('id'))
        return PromptHistory.objects.filter(id__in=[fp['first_prompt_id'] for fp in first_prompts])
    usernames_entry = User.objects.get(username=request.user.username)
    user_prompts = PromptHistory.objects.filter(user=usernames_entry)
    history_today = get_first_prompts(user_prompts.filter(created_at__date=today))
    history_yesterday = get_first_prompts(user_prompts.filter(created_at__date=yesterday))
    history_last_week = get_first_prompts(user_prompts.filter(created_at__gte=last_week, created_at__lt=yesterday))
    history_last_month = get_first_prompts(user_prompts.filter(created_at__gte=last_month, created_at__lt=last_week))
    history_data = {
        "today": PromptHistorySerializer(history_today.order_by('-created_at'), many=True).data,
        "yesterday": PromptHistorySerializer(history_yesterday.order_by('-created_at'), many=True).data,
        "last_week": PromptHistorySerializer(history_last_week.order_by('-created_at'), many=True).data,
        "last_month": PromptHistorySerializer(history_last_month.order_by('-created_at'), many=True).data
    }

    return Response(history_data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_session_history(request, session_id):
    # 1) Determine which “mode” the front end is requesting
    mode = request.GET.get('mode', 'chat')  # default to chat

    # 2) Fetch this user’s QA flag
    try:
        user_info = UserInfo.objects.get(username=request.user.username)
    except UserInfo.DoesNotExist:
        return Response({'error': 'UserInfo not found'}, status=status.HTTP_403_FORBIDDEN)

    # 3) Block non-QA users from QA mode
    if mode == 'qa' and user_info.usecase != 'qa':
        return Response({'error': 'Not authorized for QA'}, status=status.HTTP_403_FORBIDDEN)

    # 4) Build the correct history queryset
    base_qs = PromptHistory.objects.filter(
        user=request.user,
        session_id=session_id
    )
    if mode == 'qa':
        history_qs = base_qs.filter(usecase='qa')
    else:
        # chat mode: only entries with no QA flag
        history_qs = base_qs.filter(usecase__isnull=True)

    # 5) Return 404 if empty
    if not history_qs.exists():
        return Response({'error': 'No history found for this session'}, status=status.HTTP_404_NOT_FOUND)

    # 6) Serialize and return
    serializer = PromptHistorySerializer(history_qs.order_by('created_at'), many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def save_comment(request, pk):
    try:
        usernames_entry = User.objects.get(username=request.user.username)
        
        prompt_history = PromptHistory.objects.get(pk=pk, user=usernames_entry)
        comment = request.data.get('comments', '')
        if comment:
            prompt_history.comments = comment
            prompt_history.save()
            return Response({"message": "Comment saved successfully"}, status=status.HTTP_200_OK)
        else:
            return Response({"message": "No comment provided"}, status=status.HTTP_204_NO_CONTENT)

    except PromptHistory.DoesNotExist:
        return Response({"error": "Prompt history not found or unauthorized"}, status=status.HTTP_404_NOT_FOUND)
    
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def mark_satisfied_or_unsatisfied(request, pk):
    """
    Mark the feedback as unsatisfied or satisfied.
    """
    try:
     
        feedback_status = request.data.get('status')
        usernames_entry = User.objects.get(username=request.user.username)
        prompt_history = PromptHistory.objects.get(pk=pk, user=usernames_entry)
        print("Feedback_status",feedback_status)
        prompt_history.thumbs_feedback = feedback_status
        prompt_history.save()
        
        return Response({"message": "Feedback marked as unsatisfied."}, status=status.HTTP_200_OK)
    
    except PromptHistory.DoesNotExist:
        return Response({"error": "Prompt history not found or unauthorized."}, status=status.HTTP_404_NOT_FOUND)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_files(request):
    files = get_all_files_from_milvus()  
    return JsonResponse({"files":files}) 


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_documents(request):
    try:
        current_collection = CurrentUsingCollection.objects.first()  
        collection_name=current_collection.current_using_collection
        if not collection_name:
            return JsonResponse({'error': 'No current collection found'}, status=404)
        table_name = f"user_access_{collection_name}"
        with connection.cursor() as cursor:
            query = f"SELECT document_name FROM {table_name}"
            cursor.execute(query)
            documents = cursor.fetchall()
        if documents:
            file_names = [doc[0] for doc in documents]
            return JsonResponse({'files': file_names, "username":request.user.username,}, status=200)
        else:
            return JsonResponse({'error': 'No documents found for this ps_number'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_folder_name(request):
    try:
        current_collection = CurrentUsingCollection.objects.first()  
        collection_name=current_collection.current_using_collection
        if not collection_name:
            return JsonResponse({'error': 'No current collection found'}, status=404)
        table_name = f"user_access_{collection_name}"
        with connection.cursor() as cursor:
            query = f"SELECT document_name FROM {table_name}"
            cursor.execute(query)
            documents = cursor.fetchall()
        if documents:
            file_names = ["/".join(doc[0].split("/")[:-1]) for doc in documents]
            return JsonResponse({'files': file_names,}, status=200)
        else:
            return JsonResponse({'error': 'No documents found for this ps_number'}, status=404)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def serve_file(request, filename, page_number):
    try:
        filename = unquote(filename)
        file_path = None

        for root, dirs, files in os.walk(PDF_DIRECTORY):
            if filename in files:
                file_path = os.path.join(root, filename)
                break

        if not file_path:
            raise Http404("File not found")

        file_extension = os.path.splitext(filename)[1].lower()
        mime_type, _ = mimetypes.guess_type(file_path)

        if file_extension == ".docx":
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        if not mime_type:
            mime_type = "application/octet-stream"
        
        file = open(file_path, 'rb')
        response = FileResponse(file, content_type=mime_type)

        if file_extension in [".pdf", ".docx"]:
            response['Content-Disposition'] = f'inline; filename="{filename}"'
        else:
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
        return response
    except Exception as e:
        raise Http404(f"An error occurred: {str(e)}")




@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_collection_name(request):
    try:
        collections = client.list_collections()
        return JsonResponse({"collections": collections, 'username': request.user.username}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def collection_files(request, collection_name):
    if request.method == 'GET':
        try:
            connections.connect("default", host=host, port= port)
            collection = Collection(collection_name)
            iterator = collection.query_iterator(batch_size=1000, output_fields=["source"])
            results = []
            while True:
                result = iterator.next()
                if not result:
                    iterator.close()
                    break
                results.extend(result)
            results = list(set([result['source'] for result in results]))
            return JsonResponse({"results": results}, status=200)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"error": "Invalid request method"}, status=405)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated]) 
def delete_collection(request, collection_name):
    try:
        client.drop_collection(collection_name)
        connection = db_utility.create_connection()
        cursor = connection.cursor()
        table_name = f"user_access_{collection_name}"
        drop_table_query = f"DROP TABLE IF EXISTS `{table_name}`;"  
        cursor.execute(drop_table_query)
        connection.commit()
        cursor.close()
        connection.close()        
        return JsonResponse({"message": f"Collection '{collection_name}' deleted successfully."}, status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@api_view(['DELETE'])
@permission_classes([IsAuthenticated]) 
def delete_file(request, source, collection_name):
    if request.method == 'DELETE':
        try:
           
            if not source or not collection_name:
                return JsonResponse({"error": "Both 'source' and 'collection_name' are required"}, status=400)
            connections.connect("default", host=host, port= port)
            collection = Collection(collection_name)
            decoded_source = urllib.parse.unquote(source)
            delete_expr = f"source == '{decoded_source}'" 
            result = collection.delete(expr=delete_expr)
            connection = db_utility.create_connection()
            cursor = connection.cursor()
            delete_row_query = f"DELETE FROM `user_access_{collection_name}` WHERE document_name = %s;"
            cursor.execute(delete_row_query, (decoded_source,))
            connection.commit()
            cursor.close()
            connection.close()
            if result.delete_count > 0:
                return JsonResponse({"message": f"All files with source '{source}' deleted successfully"}, status=200)
            else:
                return JsonResponse({"error": "No files found with the specified source name"}, status=404)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)


def set_progress_message(message):
    """
    Update the global progress message
    """
    global progress_data
    progress_data["message"] = message


# Helper function to find files in the source folder
def find_files(root_dir, extensions):
    found_files = []
    for dirpath, _, _ in os.walk(root_dir):
        for ext in extensions:
            for filepath in glob.glob(os.path.join(dirpath, '*' + ext)):
                found_files.append(filepath)
    return found_files

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def create_collection(request):
    collection_name = request.data.get('name')
    source = request.data.get('source')
    if not collection_name or not source:
        return JsonResponse({"error": "Collection name and source are required."}, status=400)
    try:
        logger.info(f"Starting collection creation for {collection_name}.")
        start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if collection_name in client.list_collections():
            logger.info(f"Collection {collection_name} already exists. Skipping creation.")
            set_progress_message(f"Collection {collection_name} already exists. Skipping creation.")
        else:
            logger.info(f"Creating new collection: {collection_name}")
        set_progress_message(f"Creating collection: {collection_name}")

        already_chunked = db_utility.fetch_all_documents(collection_name)
        file_extensions = config('EXTENSIONS')
        found_files = find_files(source, file_extensions)

        needs_to_be_chunked = [file for file in found_files if file not in already_chunked]
        needs_to_be_chunked = list(set(needs_to_be_chunked))
        if needs_to_be_chunked:
            progress_generator = file_process_new.create_langchain_documents(needs_to_be_chunked, collection_name)

            for progress in progress_generator:
                progress_message = f"Progress: {progress['current_progress']} Files completed /  Total files  {progress['total_files']}   -   {progress['progress_percentage']:.2f}%"
                set_progress_message(progress_message)

        else:
            set_progress_message("No new files to process.")
        
        logger.info(f"File process completed.")
        # Insert into chunking monitor
        end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        insert_query_chunking_monitor = f"""
            INSERT INTO chunking_monitor (start_time, completed_time, logging_file, chunked_folder, database_name)
            VALUES ('{start_time}', '{end_time}', '{start_time}_logs.log','{collection_name}', '{source}')
        """
        db_utility.insert_chunking_monitor(insert_query_chunking_monitor)

        set_progress_message(f"Collection {collection_name} created successfully.")
        set_progress_message("Upload completed.")
        return JsonResponse({"message": "Collection creation completed."}, status=200)

    except Exception as e:
        logger.error(f"Error during collection creation: {str(e)}")
        set_progress_message(f"Error occurred: {str(e)}")
        return JsonResponse({"error": str(e)}, status=500)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_progress(request):
    return JsonResponse(progress_data)

@api_view(["GET"])
@permission_classes([IsAuthenticated]) 
def get_milvus_data(request, collection_name):
    try:
        connections.connect("default", host=host, port= port)
        page = request.GET.get("page", 1)
        try:
            page = int(page)
            if page < 1:
                return Response({"error": "Invalid page number"}, status=status.HTTP_400_BAD_REQUEST)
        except ValueError:
            return Response({"error": "Page number must be an integer"}, status=status.HTTP_400_BAD_REQUEST)

        page_size = 150
        offset = (page - 1) * page_size

        try:
            collection = Collection(name=collection_name)
            collection.load()
        except Exception as e:
            return Response({"error": f"Failed to load collection: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            results = collection.query(
                expr="",  
                output_fields=["source", "page", "text", "pk"],
                offset=offset,
                limit=page_size,
            )

            if not results:
                return Response({
                    "data": [],
                    "page": page,
                    "page_size": page_size,
                    "next_page": None 
                }, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": f"Error querying Milvus: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        next_page = page + 1 if len(results) == page_size else None

        response_data = {
            "data": results,
            "page": page,
            "page_size": page_size,
            "next_page": next_page
        }
        return JsonResponse(response_data, safe=False)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)


@api_view(["GET"])
@permission_classes([IsAuthenticated]) 
def get_current_using_collection(request):
    try:
        current_collection = CurrentUsingCollection.objects.first() 
        if current_collection:
            serializer = CurrentUsingCollectionSerializer(current_collection)
        
            return Response(serializer.data)
        
        else:
            return Response({"message": "No collection selected"}, status=404)
    except Exception as e:
        return Response({"error": f"Error fetching current collection: {str(e)}"}, status=500)
    


@api_view(["POST"])
@permission_classes([IsAuthenticated]) 
def update_current_collection(request):

    current_collection = request.data.get("current_using_collection")
    if not current_collection:
        return Response({"error": "No collection selected"}, status=400)

    current_collection_instance, created = CurrentUsingCollection.objects.update_or_create(
        defaults={"current_using_collection": current_collection}
    )
    
    global_collection_name = current_collection_instance.current_using_collection
    return Response({
        "message": "Collection updated successfully", 
        "current_using_collection": global_collection_name
    }, status=200)



@api_view(["GET"])
@permission_classes([IsAuthenticated]) 
def restart_server(request):
    if request.method == 'GET':
        try:
            subprocess.Popen(['python3', 'manage.py', 'runserver', '0.0.0.0:8000'])

            pid = os.getpid()
            os.kill(pid, 9) 

            return JsonResponse({'message': 'Server restarting...'})
        except Exception as e:
            return JsonResponse({'message': f'Error: {str(e)}'}, status=500)
    else:
        return JsonResponse({'message': 'Method not allowed'}, status=405)