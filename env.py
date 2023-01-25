import os
from dotenv import load_dotenv


class Env:
    """
    Class used to get environment variables
    """
    HOST_MYSQL: str
    USER_MYSQL: str
    PWD_MYSQL: str
    DB_MYSQL: str

    def __init__(self) -> None:
        load_dotenv()
        self.HOST_MYSQL = os.getenv('HOST_MYSQL', '')
        self.USER_MYSQL = os.getenv('USER_MYSQL', '')
        self.PWD_MYSQL = os.getenv('PWD_MYSQL', '')
        self.DB_MYSQL = os.getenv('DB_MYSQL', '')
