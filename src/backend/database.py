from pymongo import AsyncMongoClient
from core.config import settings

class Database:
    client: AsyncMongoClient = None

db = Database()

def get_database():
    return db.client[settings.DATABASE_NAME]
