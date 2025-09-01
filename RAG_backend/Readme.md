### Commands to make

To set up and run the Django project, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up the environment variables:**
    Create a `.env` file in the root directory and add the necessary environment variables as specified in the project documentation.

5. **Apply the migrations:**
    ```bash
    python manage.py makemigrations
    python manage.py migrate
    ```

6. **Run the development server:**
    ```bash
    python manage.py runserver
    ```

7. **Create a superuser (optional):**
    ```bash
    python manage.py createsuperuser
    ```

8. **Run tests (optional):**
    ```bash
    python manage.py test
    ```

Make sure to replace `<repository_url>` and `<repository_name>` with the actual repository URL and name.