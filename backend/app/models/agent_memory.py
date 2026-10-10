from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)

from backend.app.db.database import Base

#这是一个 SQLAlchemy ORM 数据库模型AgentMemory，定义了 agent_memories 表，用于存储 Agent 的长期记忆。
#SQLAlchemy ORM 是 Python 中一种“用对象操作数据库”的方式。
#ORM 全称是 Object-Relational Mapping，即“对象—关系映射”。
#把数据库表映射为 Python 类，把表中的记录映射为类的实例，把表中的字段映射为类的属性。


#可以存储用户级别的记忆（memory_scope = 'user'）或工作空间级别的记忆（memory_scope = 'workspace'）。

class AgentMemory(Base):#继承 Base 表示这是一个数据库映射类。
    #Python 的 AgentMemory 对象 对应 PostgreSQL 的 agent_memories 表
    __tablename__ = "agent_memories"

    __table_args__ = (
        CheckConstraint(#用户记忆必须明确属于一个用户；工作空间共享记忆不能挂在某个用户名下。
            """
            (
                memory_scope = 'user'
                AND user_id IS NOT NULL
            )
            OR
            (
                memory_scope = 'workspace'
                AND user_id IS NULL
            )
            """,
            name="ck_agent_memories_scope_user",
        ),

        Index(#用户记忆的唯一索引
            "uq_agent_memories_user",
            "workspace_id",
            "user_id",
            "memory_key",
            unique=True,
            postgresql_where=text(
                "memory_scope = 'user'"
            ),
        ),

        Index(#工作空间共享记忆的唯一索引
            "uq_agent_memories_workspace",
            "workspace_id",
            "memory_key",
            unique=True,
            postgresql_where=text(
                "memory_scope = 'workspace'"
            ),
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    workspace_id = Column(
        Integer,
        ForeignKey("workspaces.id"),
        nullable=False,
        index=True,
    )

    memory_scope = Column(
        String(20),
        nullable=False,
        index=True,
    )

    user_id = Column(
        String(100),
        nullable=True,
        index=True,
    )

    memory_key = Column(
        String(100),
        nullable=False,
    )

    memory_value = Column(
        Text,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )
