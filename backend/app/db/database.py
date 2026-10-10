from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.app.core.config import DATABASE_URL


engine = create_engine(#链接数据库
    DATABASE_URL,
    echo=True,
    pool_pre_ping=True,
)


SessionLocal = sessionmaker(#创建每次操作用的数据库会话
    autocommit=False,
    autoflush=False,
    bind=engine,
)


Base = declarative_base()


def get_db():#给每个 HTTP 请求自动提供并关闭会话
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()