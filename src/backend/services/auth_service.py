from typing import Optional
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime
from bson import ObjectId

from models.user import UserCreate, UserInDB, TokenData, User
from core.security import get_password_hash, verify_password, create_access_token
from core.config import settings
from database import get_database

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

async def get_user_by_email(email: str) -> Optional[dict]:
    db = get_database()
    user = await db["users"].find_one({"email": email})
    return user

async def get_user_by_id(user_id: str) -> Optional[dict]:
    db = get_database()
    try:
        user = await db["users"].find_one({"_id": ObjectId(user_id)})
        return user
    except Exception:
        return None

async def create_user(user: UserCreate) -> dict:
    db = get_database()
    existing_user = await get_user_by_email(user.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    user_dict = user.model_dump()
    user_dict["hashed_password"] = hashed_password
    del user_dict["password"]
    user_dict["created_at"] = datetime.utcnow()
    
    result = await db["users"].insert_one(user_dict)
    created_user = await get_user_by_id(str(result.inserted_id))
    if created_user:
        created_user["_id"] = str(created_user["_id"])
    return created_user

async def update_user(email: str, user_update) -> dict:
    db = get_database()
    update_data = {}
    if user_update.full_name is not None:
        update_data["full_name"] = user_update.full_name
    if user_update.personal_context is not None:
        update_data["personal_context"] = user_update.personal_context.strip()
    if user_update.password is not None and user_update.password != "":
        update_data["hashed_password"] = get_password_hash(user_update.password)
        
    if update_data:
        await db["users"].update_one({"email": email}, {"$set": update_data})
        
    updated_user = await get_user_by_email(email)
    updated_user["_id"] = str(updated_user["_id"])
    return updated_user

async def authenticate_user(email: str, password: str) -> Optional[dict]:
    user = await get_user_by_email(email)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    user = await get_user_by_email(token_data.email)
    if user is None:
        raise credentials_exception
    
    user["_id"] = str(user["_id"])
    return User(**user)
