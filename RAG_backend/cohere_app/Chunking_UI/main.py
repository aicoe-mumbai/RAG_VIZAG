import os
import datetime
import glob
import streamlit as st
import pandas as pd
from RAG_backend.cohere_app.Chunking_UI.file_process import create_langchain_documents, ERROR_FOLDER
from pymilvus import MilvusClient
from enable_logging import logger, current_time
import db_utility
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()
Milvus_url = os.getenv("MILVUS_URL")

def check():
    return db_utility.fetch_all_documents()

def find_files(root_dir, extensions):
    found_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for ext in extensions:
            for filepath in glob.glob(os.path.join(dirpath, '*' + ext)):
                found_files.append(filepath)
    return found_files

# logger = logging.getLogger(__name__)
chunking_monitor = db_utility.chunking_monitor()
if(chunking_monitor == False):
    logger.ERROR("Error found at creation of chunking monitor table")

client = MilvusClient(
    uri=Milvus_url,
    token="root:Milvus"
)
# connection and database table creation command
create_error_files_table = db_utility.create_error_files()

if(create_error_files_table == False):
    logger.error("Found error in creating error files table : ")
    

collection_name = input("Type the collection name : ")
source_folder = input("Folder path : ")
create_user_access_table = db_utility.create_user_access(collection_name)

if(create_user_access_table == False):
    logger.error("Error encountered at creating the user access table")

    file_extensions = ['.PDF','.pdf', '.txt', '.pptx', '.docx', '.xlsx', '.csv'] 
    found_files = find_files(source_folder, file_extensions)
    print(found_files)
    if(os.path.exists(source_folder)):
        if collection_name in client.list_collections():
            already_chunked = check()
            needs_to_be_chunked = [item for item in found_files if item not in already_chunked]
        else:
            needs_to_be_chunked = found_files
        print("Already", needs_to_be_chunked)
        if needs_to_be_chunked:
            create_langchain_documents(needs_to_be_chunked, collection_name)
end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
insert_query_chunking_monitor = f"""
    INSERT INTO chunking_monitor (start_time, completed_time, logging_file, chunked_folder, database_name)
    VALUES 
    ('{current_time}', '{end_time}', '{current_time}_logs.log','{source_folder}', '{collection_name}')
"""

db_utility.insert_chunking_monitor(insert_query_chunking_monitor)


error_files = []
processed_files = []
exit()