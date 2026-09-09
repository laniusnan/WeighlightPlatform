import enum
import uuid
from datetime import UTC, datetime
from typing import Any

from pydantic import EmailStr
from sqlalchemy import JSON, BigInteger, Column, DateTime
from sqlmodel import Field, Relationship, SQLModel


def get_datetime_utc() -> datetime:
    return datetime.now(UTC)


# Framework of the original model
class ModelFramework(enum.StrEnum):
    pytorch = "pytorch"
    onnx = "onnx"
    huggingface = "huggingface"
    tensorflow = "tensorflow"


# How the model got into the platform
class ModelSourceType(enum.StrEnum):
    upload = "upload"
    url = "url"
    # Imported from a directory on the server host (mounted read-only into the
    # container, e.g. /home/zhangxuefei/Model -> /models-local).
    local = "local"


class ModelStatus(enum.StrEnum):
    pending = "pending"
    ready = "ready"
    failed = "failed"


# Supported compression methods (all CPU-friendly)
class CompressionMethod(enum.StrEnum):
    export_onnx = "export_onnx"  # PyTorch -> ONNX (+ onnxsim simplify)
    quantize_onnx = "quantize_onnx"  # ONNX dynamic INT8 quantization
    quantize_onnx_static = "quantize_onnx_static"  # ONNX static INT8 (with calibration data)
    openvino_int8 = "openvino_int8"  # OpenVINO IR + NNCF INT8 post-training quantization
    prune = "prune"  # L1/L2/BN-Slim structured filter pruning (PyTorch CNN)
    sparsify = "sparsify"  # Unstructured magnitude pruning (uniform / global threshold)
    distill = "distill"  # Knowledge distillation (teacher -> width_mult student, per-class images)
    qat = "qat"  # Quantization-aware fine-tuning (PyTorch FX / LSQ) -> quantized ONNX
    llm_finetune = "llm_finetune"  # CPU micro LoRA SFT prototype (GPU phase II: XTuner)
    export_gguf = "export_gguf"  # HuggingFace LLM -> GGUF (llama.cpp) for llama.cpp tools
    llm_quantize = "llm_quantize"  # LLM INT4/INT8 via optimum-intel (OpenVINO IR)


class TaskStatus(enum.StrEnum):
    pending = "pending"
    running = "running"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"


# Shared properties
class UserBase(SQLModel):
    email: EmailStr = Field(unique=True, index=True, max_length=255)
    is_active: bool = True
    is_superuser: bool = False
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)


class UserRegister(SQLModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = Field(default=None, max_length=255)


# Properties to receive via API on update, all are optional
class UserUpdate(SQLModel):
    email: EmailStr | None = Field(default=None, max_length=255)
    is_active: bool | None = None
    is_superuser: bool | None = None
    full_name: str | None = Field(default=None, max_length=255)
    password: str | None = Field(default=None, min_length=8, max_length=128)


class UserUpdateMe(SQLModel):
    full_name: str | None = Field(default=None, max_length=255)
    email: EmailStr | None = Field(default=None, max_length=255)


class UpdatePassword(SQLModel):
    current_password: str = Field(min_length=8, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


# Database model, database table inferred from class name
class User(UserBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hashed_password: str
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    items: list["Item"] = Relationship(back_populates="owner", cascade_delete=True)
    models: list["Model"] = Relationship(back_populates="owner", cascade_delete=True)
    datasets: list["Dataset"] = Relationship(back_populates="owner", cascade_delete=True)
    compression_tasks: list["CompressionTask"] = Relationship(
        back_populates="owner", cascade_delete=True
    )


# Properties to return via API, id is always required
class UserPublic(UserBase):
    id: uuid.UUID
    created_at: datetime | None = None


class UsersPublic(SQLModel):
    data: list[UserPublic]
    count: int


# Shared properties
class ItemBase(SQLModel):
    title: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Properties to receive on item creation
class ItemCreate(ItemBase):
    pass


# Properties to receive on item update
class ItemUpdate(SQLModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=255)


# Database model, database table inferred from class name
class Item(ItemBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="items")


# Properties to return via API, id is always required
class ItemPublic(ItemBase):
    id: uuid.UUID
    owner_id: uuid.UUID
    created_at: datetime | None = None


class ItemsPublic(SQLModel):
    data: list[ItemPublic]
    count: int


# Generic message
class Message(SQLModel):
    message: str


# JSON payload containing access token
class Token(SQLModel):
    access_token: str
    token_type: str = "bearer"


# Contents of JWT token
class TokenPayload(SQLModel):
    sub: str | None = None


class NewPassword(SQLModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)


# Shared properties
class ModelBase(SQLModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=1024)
    # Model registry: version + tags (comma separated)
    version: int = Field(default=1)
    tags: str | None = Field(default=None, max_length=255)


# Properties to receive on model creation (upload or URL pull)
class ModelCreate(ModelBase):
    framework: ModelFramework = ModelFramework.pytorch
    source_url: str | None = Field(default=None, max_length=1024)


# Database model, table inferred from class name
class Model(ModelBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # Stored as a plain string in the DB; the API schemas expose the enum.
    framework: str = Field(default=ModelFramework.pytorch.value, max_length=32)
    source_type: str = Field(default=ModelSourceType.upload.value, max_length=32)
    source_url: str | None = Field(default=None, max_length=1024)
    status: str = Field(default=ModelStatus.pending.value, max_length=32, index=True)
    file_path: str | None = Field(default=None, max_length=1024)
    size_bytes: int = Field(default=0, sa_column=Column(BigInteger))
    error_message: str | None = Field(default=None, max_length=2048)
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="models")
    compression_tasks: list["CompressionTask"] = Relationship(
        back_populates="model", cascade_delete=True
    )


# Properties to return via API
class ModelPublic(ModelBase):
    id: uuid.UUID
    framework: ModelFramework
    source_type: ModelSourceType
    source_url: str | None
    status: ModelStatus
    size_bytes: int
    error_message: str | None = None
    created_at: datetime | None = None


class ModelsPublic(SQLModel):
    data: list[ModelPublic]
    count: int


# Properties to receive on compression task creation
class CompressionTaskCreate(SQLModel):
    model_id: uuid.UUID
    method: CompressionMethod
    params: dict[str, Any] = Field(default_factory=dict)


# Database model, table inferred from class name
class CompressionTask(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    model_id: uuid.UUID = Field(
        foreign_key="model.id", nullable=False, ondelete="CASCADE", index=True
    )
    method: str = Field(max_length=64, index=True)
    params: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    status: str = Field(default=TaskStatus.pending.value, max_length=32, index=True)
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    error_message: str | None = Field(default=None, max_length=2048)
    output_path: str | None = Field(default=None, max_length=1024)
    output_size_bytes: int | None = Field(default=None, sa_column=Column(BigInteger))
    input_size_bytes: int | None = Field(default=None, sa_column=Column(BigInteger))
    metrics: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    started_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)  # type: ignore
    )
    finished_at: datetime | None = Field(
        default=None, sa_type=DateTime(timezone=True)  # type: ignore
    )
    model: Model | None = Relationship(back_populates="compression_tasks")
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE", index=True
    )
    owner: User | None = Relationship(back_populates="compression_tasks")


# Properties to return via API
class CompressionTaskPublic(SQLModel):
    id: uuid.UUID
    model_id: uuid.UUID
    model_name: str | None = None
    method: CompressionMethod
    params: dict[str, Any]
    status: TaskStatus
    progress: float
    error_message: str | None
    output_size_bytes: int | None
    input_size_bytes: int | None
    metrics: dict[str, Any]
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class CompressionTasksPublic(SQLModel):
    data: list[CompressionTaskPublic]
    count: int


# Shared properties
class DatasetBase(SQLModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=1024)
    source_type: str = Field(default="upload", max_length=32)
    source_url: str | None = Field(default=None, max_length=1024)
    status: str = Field(default="ready", max_length=32)


# Database model, table inferred from class name
class Dataset(DatasetBase, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    # Local path to the extracted dataset directory (or single file).
    file_path: str | None = Field(default=None, max_length=1024)
    size_bytes: int = Field(default=0, sa_column=Column(BigInteger))
    created_at: datetime | None = Field(
        default_factory=get_datetime_utc,
        sa_type=DateTime(timezone=True),  # type: ignore
    )
    owner_id: uuid.UUID = Field(
        foreign_key="user.id", nullable=False, ondelete="CASCADE"
    )
    owner: User | None = Relationship(back_populates="datasets")


# Properties to return via API
class DatasetPublic(DatasetBase):
    id: uuid.UUID
    size_bytes: int
    created_at: datetime | None = None


class DatasetsPublic(SQLModel):
    data: list[DatasetPublic]
    count: int
