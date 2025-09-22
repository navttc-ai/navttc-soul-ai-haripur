# Advanced Object-Oriented Programming Concepts in Python

A comprehensive guide to mastering advanced OOP concepts in Python with practical examples, assignments, and learning resources.

## Table of Contents

1. [Polymorphism](#1-polymorphism)
2. [Operator Overloading](#2-operator-overloading)
3. [Magic (Dunder) Methods](#3-magic-dunder-methods)
4. [Dynamic Polymorphism](#4-dynamic-polymorphism)
5. [Abstract Classes and Methods](#5-abstract-classes-and-methods)
6. [Empty Classes](#6-empty-classes)
7. [Data Classes](#7-data-classes)
8. [Keyword Arguments](#8-keyword-arguments)
9. [Additional Resources](#additional-resources)
10. [Practice Exercises](#practice-exercises)

---

## 1. Polymorphism

### Definition and Concept

Polymorphism, derived from Greek meaning "many forms," is a fundamental principle in object-oriented programming that allows a single interface to represent different underlying forms (data types). This means the same method call can behave differently depending on the object it is called upon.

### Key Benefits

- **Code Reusability**: Write functions that work with multiple object types
- **Flexibility**: Easy to extend code with new types without modifying existing code
- **Maintainability**: Reduces code duplication and complexity

### Basic Example

```python
# Different classes with the same method name
class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Bird:
    def speak(self):
        return "Tweet!"

# Polymorphic function - works with any object that has a speak() method
def make_animal_speak(animal):
    return animal.speak()

# Usage
move_object(player, 5, 0, 0)
render_object(player)
```

### Configuration Management Example

```python
class AppConfig:
    pass

class DatabaseSettings:
    pass

class APISettings:
    pass

# Create hierarchical configuration
config = AppConfig()

# Database configuration
config.database = DatabaseSettings()
config.database.host = "localhost"
config.database.port = 5432
config.database.name = "production_db"
config.database.pool_size = 10

# API configuration
config.api = APISettings()
config.api.base_url = "https://api.example.com"
config.api.timeout = 30
config.api.retry_attempts = 3
config.api.rate_limit = 1000

# Application settings
config.debug = False
config.log_level = "INFO"
config.secret_key = "your-secret-key"

def load_config_from_dict(config_obj, config_dict):
    """Dynamically load configuration from a dictionary"""
    for key, value in config_dict.items():
        if isinstance(value, dict):
            # Create nested empty class for nested dictionaries
            nested_config = type(f"{key.capitalize()}Config", (), {})()
            setattr(config_obj, key, nested_config)
            load_config_from_dict(nested_config, value)
        else:
            setattr(config_obj, key, value)

# Load from dictionary
config_data = {
    "database": {
        "host": "db.example.com",
        "port": 3306,
        "credentials": {
            "username": "app_user",
            "password": "secure_password"
        }
    },
    "cache": {
        "redis_url": "redis://localhost:6379",
        "ttl": 3600
    }
}

dynamic_config = AppConfig()
load_config_from_dict(dynamic_config, config_data)

print(f"Database host: {dynamic_config.database.host}")
print(f"Cache TTL: {dynamic_config.cache.ttl}")
```

---

## 7. Data Classes

### Introduction and Benefits

Data classes, introduced in Python 3.7, provide a decorator that automatically generates common methods like `__init__`, `__repr__`, `__eq__`, and others. They're designed for classes that primarily store data with minimal behavior.

### Key Advantages

- **Reduced Boilerplate**: Automatic generation of common methods
- **Type Hints Integration**: Built-in support for type annotations
- **Immutability Options**: Support for frozen (immutable) data classes
- **Default Values**: Easy specification of default values
- **Comparison Methods**: Automatic generation of comparison operations

### Basic Data Class Features

```python
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

@dataclass
class Person:
    name: str
    age: int
    email: str
    active: bool = True  # Default value
    
# Automatically generates __init__, __repr__, __eq__, etc.
person1 = Person("Alice", 30, "alice@example.com")
person2 = Person("Bob", 25, "bob@example.com", False)

print(person1)  # Person(name='Alice', age=30, email='alice@example.com', active=True)
print(person1 == person2)  # False
```

### Advanced Data Class Example

```python
from dataclasses import dataclass, field, asdict, astuple
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass(order=True)  # Enable comparison operations
class Task:
    title: str
    description: str
    priority: Priority
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Custom field that's not included in comparison
    id: str = field(default_factory=lambda: f"task_{datetime.now().timestamp()}", 
                   compare=False)
    
    def __post_init__(self):
        """Called after __init__, useful for validation or derived fields"""
        if self.due_date and self.due_date < self.created_at:
            raise ValueError("Due date cannot be before creation date")
        
        # Add automatic tagging based on priority
        if self.priority == Priority.CRITICAL and "urgent" not in self.tags:
            self.tags.append("urgent")
    
    @property
    def is_overdue(self) -> bool:
        if not self.due_date or self.completed:
            return False
        return datetime.now() > self.due_date
    
    def add_tag(self, tag: str) -> None:
        if tag not in self.tags:
            self.tags.append(tag)
    
    def complete(self) -> None:
        self.completed = True
        self.metadata['completed_at'] = datetime.now().isoformat()

@dataclass(frozen=True)  # Immutable data class
class ProjectInfo:
    name: str
    description: str
    owner: str
    created_at: datetime = field(default_factory=datetime.now)
    
    def __str__(self):
        return f"Project: {self.name} (Owner: {self.owner})"

@dataclass
class Project:
    info: ProjectInfo
    tasks: List[Task] = field(default_factory=list)
    
    def add_task(self, task: Task) -> None:
        self.tasks.append(task)
    
    def get_tasks_by_priority(self, priority: Priority) -> List[Task]:
        return [task for task in self.tasks if task.priority == priority]
    
    def get_overdue_tasks(self) -> List[Task]:
        return [task for task in self.tasks if task.is_overdue]
    
    def completion_rate(self) -> float:
        if not self.tasks:
            return 0.0
        completed_tasks = sum(1 for task in self.tasks if task.completed)
        return completed_tasks / len(self.tasks)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'info': asdict(self.info),
            'tasks': [asdict(task) for task in self.tasks],
            'stats': {
                'total_tasks': len(self.tasks),
                'completion_rate': self.completion_rate(),
                'overdue_tasks': len(self.get_overdue_tasks())
            }
        }

# Usage example
project_info = ProjectInfo(
    name="Website Redesign",
    description="Complete overhaul of company website",
    owner="Alice Johnson"
)

project = Project(info=project_info)

# Add tasks
tasks = [
    Task("Design mockups", "Create initial design concepts", Priority.HIGH,
         tags=["design", "frontend"]),
    Task("Backend API", "Develop REST API endpoints", Priority.MEDIUM,
         tags=["backend", "api"]),
    Task("Database schema", "Design and implement database", Priority.HIGH,
         tags=["database", "backend"])
]

for task in tasks:
    project.add_task(task)

# Work with tasks
high_priority_tasks = project.get_tasks_by_priority(Priority.HIGH)
print(f"High priority tasks: {len(high_priority_tasks)}")

# Complete a task
tasks[0].complete()
print(f"Completion rate: {project.completion_rate():.2%}")

# Convert to dictionary for JSON serialization
project_dict = project.to_dict()
print(f"Project data: {project_dict['stats']}")
```

### Data Class with Validation

```python
from dataclasses import dataclass, field
from typing import ClassVar
import re

@dataclass
class User:
    username: str
    email: str
    age: int
    
    # Class variable (not an instance field)
    _email_pattern: ClassVar[str] = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}
animals = [Dog(), Cat(), Bird()]
for animal in animals:
    print(make_animal_speak(animal))
```

### Advanced Example: AI Model Interface

```python
class AIModel:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def predict(self, input_data):
        raise NotImplementedError("Subclasses must implement predict method")

class ImageClassifier(AIModel):
    def predict(self, image_data):
        # Simulate image classification
        print(f"{self.model_name}: Analyzing image...")
        return "Object detected: Cat"

class TextAnalyzer(AIModel):
    def predict(self, text_data):
        # Simulate text analysis
        print(f"{self.model_name}: Analyzing text sentiment...")
        return "Sentiment: Positive"

class VoiceRecognizer(AIModel):
    def predict(self, audio_data):
        # Simulate voice recognition
        print(f"{self.model_name}: Processing audio...")
        return "Transcription: Hello World"

# Polymorphic usage
def process_data(model, data):
    return model.predict(data)

models = [
    ImageClassifier("ResNet-50"),
    TextAnalyzer("BERT"),
    VoiceRecognizer("Whisper")
]

sample_data = ["image.jpg", "Great product!", "audio.wav"]

for i, model in enumerate(models):
    result = process_data(model, sample_data[i])
    print(f"Result: {result}\n")
```

### Assignment: Vehicle Management System

Create a polymorphic vehicle management system:

```python
# Your task: Implement the following classes
class Vehicle:
    def start_engine(self):
        pass
    
    def get_info(self):
        pass

class Car(Vehicle):
    # Implement start_engine and get_info methods
    pass

class Motorcycle(Vehicle):
    # Implement start_engine and get_info methods
    pass

class Bicycle(Vehicle):
    # Implement start_engine and get_info methods
    pass
```

---

## 2. Operator Overloading

### Definition and Purpose

Operator overloading is a specific form of polymorphism that allows you to define custom behaviors for built-in operators (`+`, `-`, `*`, `==`, etc.) when used with your custom objects. This is achieved through special "dunder" (double underscore) methods.

### Common Operator Methods

| Operator | Method | Description |
|----------|---------|-------------|
| `+` | `__add__(self, other)` | Addition |
| `-` | `__sub__(self, other)` | Subtraction |
| `*` | `__mul__(self, other)` | Multiplication |
| `/` | `__truediv__(self, other)` | Division |
| `==` | `__eq__(self, other)` | Equality |
| `<` | `__lt__(self, other)` | Less than |
| `>` | `__gt__(self, other)` | Greater than |
| `len()` | `__len__(self)` | Length |

### Basic Example: Money Class

```python
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot add {self.currency} and {other.currency}")
            return Money(self.amount + other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount + other, self.currency)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
            return Money(self.amount - other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount - other, self.currency)
        return NotImplemented
    
    def __mul__(self, multiplier):
        if isinstance(multiplier, (int, float)):
            return Money(self.amount * multiplier, self.currency)
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Money):
            return self.amount == other.amount and self.currency == other.currency
        return False
    
    def __lt__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
            return self.amount < other.amount
        return NotImplemented
    
    def __repr__(self):
        return f"Money({self.amount}, '{self.currency}')"
    
    def __str__(self):
        return f"{self.amount} {self.currency}"

# Usage examples
wallet1 = Money(100, "USD")
wallet2 = Money(50, "USD")
cash = Money(25, "USD")

total = wallet1 + wallet2 + cash
print(total)  # 175 USD

doubled = wallet1 * 2
print(doubled)  # 200 USD

print(wallet1 > wallet2)  # True
```

### Advanced Example: Vector Class

```python
import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)) and scalar != 0:
            return Vector(self.x / scalar, self.y / scalar)
        return NotImplemented
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
    
    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def __str__(self):
        return f"Vector({self.x:.2f}, {self.y:.2f})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # Dot product using @ operator
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

# Usage
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4.00, 6.00)
print(v1 * 2)       # Vector(6.00, 8.00)
print(abs(v1))      # 5.0
print(v1 @ v2)      # 11 (dot product)
```

### Assignment: Complex Number Class

Create a comprehensive complex number class with operator overloading.

---

## 3. Magic (Dunder) Methods

### Overview

Magic methods (also called dunder methods) are special methods in Python that start and end with double underscores. They allow your objects to integrate seamlessly with Python's built-in functions and operators.

### Categories of Magic Methods

#### 1. Object Representation
- `__str__(self)`: Human-readable string representation
- `__repr__(self)`: Developer-friendly representation
- `__format__(self, format_spec)`: Custom formatting

#### 2. Numeric Operations
- `__add__`, `__sub__`, `__mul__`, etc.: Arithmetic operations
- `__iadd__`, `__isub__`, `__imul__`, etc.: In-place operations

#### 3. Comparison Operations
- `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`: Comparisons

#### 4. Container Emulation
- `__len__(self)`: Length of container
- `__getitem__(self, key)`: Item access with `obj[key]`
- `__setitem__(self, key, value)`: Item assignment
- `__delitem__(self, key)`: Item deletion
- `__contains__(self, item)`: Membership testing with `in`
- `__iter__(self)`: Iterator protocol

#### 5. Callable Objects
- `__call__(self, *args, **kwargs)`: Make objects callable like functions

### Comprehensive Example: Smart Dictionary

```python
class SmartDict:
    def __init__(self, data=None):
        self._data = data or {}
        self._access_count = {}
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found")
        self._access_count[key] = self._access_count.get(key, 0) + 1
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        if key not in self._access_count:
            self._access_count[key] = 0
    
    def __delitem__(self, key):
        if key in self._data:
            del self._data[key]
            del self._access_count[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def __str__(self):
        return str(self._data)
    
    def __repr__(self):
        return f"SmartDict({self._data})"
    
    def __call__(self, key):
        """Make the dictionary callable to get access count"""
        return self._access_count.get(key, 0)
    
    def most_accessed(self):
        if not self._access_count:
            return None
        return max(self._access_count.items(), key=lambda x: x[1])

# Usage
smart_dict = SmartDict({'a': 1, 'b': 2, 'c': 3})

print(len(smart_dict))    # 3
print(smart_dict['a'])    # 1
print('a' in smart_dict)  # True

smart_dict['d'] = 4
del smart_dict['b']

for key in smart_dict:
    print(f"{key}: {smart_dict[key]}")

print(smart_dict('a'))    # Access count for 'a'
print(smart_dict.most_accessed())  # Most accessed key
```

### File-like Object Example

```python
class MemoryFile:
    def __init__(self, initial_content=""):
        self._content = initial_content
        self._position = 0
        self._closed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self):
        return len(self._content)
    
    def __str__(self):
        return self._content
    
    def write(self, text):
        if self._closed:
            raise ValueError("Cannot write to closed file")
        self._content += text
        return len(text)
    
    def read(self, size=-1):
        if self._closed:
            raise ValueError("Cannot read from closed file")
        if size == -1:
            result = self._content[self._position:]
            self._position = len(self._content)
        else:
            result = self._content[self._position:self._position + size]
            self._position += len(result)
        return result
    
    def close(self):
        self._closed = True

# Usage with context manager
with MemoryFile() as f:
    f.write("Hello, World!")
    print(len(f))  # 13
    print(str(f))  # Hello, World!
```

---

## 4. Dynamic Polymorphism

### Concept and Implementation

Dynamic polymorphism occurs when the specific method to be executed is determined at runtime based on the object's actual type. This is achieved through inheritance and method overriding, allowing you to write flexible code that works with entire families of objects.

### Benefits of Dynamic Polymorphism

- **Extensibility**: Easy to add new types without modifying existing code
- **Maintainability**: Central logic handles all subtypes uniformly
- **Code Reduction**: Eliminates complex conditional logic

### Real-World Example: Payment Processing System

```python
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

class PaymentMethod(ABC):
    def __init__(self):
        self.transaction_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
    
    @abstractmethod
    def process_payment(self, amount, currency="USD"):
        pass
    
    @abstractmethod
    def validate_payment_info(self):
        pass
    
    def get_transaction_details(self):
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'method': self.__class__.__name__
        }

class CreditCard(PaymentMethod):
    def __init__(self, card_number, cvv, expiry_date):
        super().__init__()
        self.card_number = self._mask_card_number(card_number)
        self.cvv = cvv
        self.expiry_date = expiry_date
    
    def _mask_card_number(self, card_number):
        return f"****-****-****-{card_number[-4:]}"
    
    def validate_payment_info(self):
        # Simulate validation logic
        return len(self.cvv) == 3 and self.expiry_date > datetime.now()
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid card information"}
        
        # Simulate payment processing
        print(f"Processing ${amount} {currency} via Credit Card {self.card_number}")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

class PayPal(PaymentMethod):
    def __init__(self, email, password):
        super().__init__()
        self.email = email
        self._password_hash = hash(password)  # In reality, use proper hashing
    
    def validate_payment_info(self):
        return "@" in self.email and "." in self.email
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid PayPal credentials"}
        
        print(f"Processing ${amount} {currency} via PayPal ({self.email})")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

class BankTransfer(PaymentMethod):
    def __init__(self, account_number, routing_number, bank_name):
        super().__init__()
        self.account_number = f"****{account_number[-4:]}"
        self.routing_number = routing_number
        self.bank_name = bank_name
    
    def validate_payment_info(self):
        return len(self.routing_number) == 9
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid bank information"}
        
        print(f"Processing ${amount} {currency} via Bank Transfer ({self.bank_name})")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

# Payment processor that works with any payment method
class PaymentProcessor:
    def __init__(self):
        self.transaction_history = []
    
    def process_transaction(self, payment_method: PaymentMethod, amount: float):
        """This method works with any payment method - that's polymorphism!"""
        result = payment_method.process_payment(amount)
        self.transaction_history.append(result)
        return result
    
    def get_successful_transactions(self):
        return [t for t in self.transaction_history if t.get("status") == "success"]

# Usage demonstration
processor = PaymentProcessor()

# Different payment methods
payment_methods = [
    CreditCard("1234567890123456", "123", datetime(2025, 12, 31)),
    PayPal("user@example.com", "password123"),
    BankTransfer("1234567890", "123456789", "Example Bank")
]

# Process payments polymorphically
amounts = [99.99, 149.50, 299.99]

for i, method in enumerate(payment_methods):
    result = processor.process_transaction(method, amounts[i])
    print(f"Transaction result: {result['status']}")
    print("-" * 50)

print(f"Successful transactions: {len(processor.get_successful_transactions())}")
```

### Media Player Example

```python
class MediaPlayer(ABC):
    @abstractmethod
    def play(self):
        pass
    
    @abstractmethod
    def pause(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass

class AudioPlayer(MediaPlayer):
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.is_playing = False
    
    def play(self):
        self.is_playing = True
        print(f"Playing audio: {self.audio_file}")
    
    def pause(self):
        self.is_playing = False
        print(f"Paused audio: {self.audio_file}")
    
    def stop(self):
        self.is_playing = False
        print(f"Stopped audio: {self.audio_file}")

class VideoPlayer(MediaPlayer):
    def __init__(self, video_file):
        self.video_file = video_file
        self.is_playing = False
    
    def play(self):
        self.is_playing = True
        print(f"Playing video: {self.video_file}")
    
    def pause(self):
        self.is_playing = False
        print(f"Paused video: {self.video_file}")
    
    def stop(self):
        self.is_playing = False
        print(f"Stopped video: {self.video_file}")

# Polymorphic media controller
def control_media(player: MediaPlayer, action: str):
    if action == "play":
        player.play()
    elif action == "pause":
        player.pause()
    elif action == "stop":
        player.stop()

# Usage
players = [
    AudioPlayer("song.mp3"),
    VideoPlayer("movie.mp4")
]

for player in players:
    control_media(player, "play")
    control_media(player, "pause")
```

---

## 5. Abstract Classes and Methods

### Purpose and Benefits

Abstract classes serve as blueprints that cannot be instantiated directly. They define a contract that subclasses must follow by implementing all abstract methods. This ensures consistency across related classes and helps prevent errors during development.

### Key Features

- **Contract Enforcement**: Subclasses must implement all abstract methods
- **Documentation**: Abstract classes document the expected interface
- **Polymorphism Support**: Enable polymorphic behavior across implementations
- **Code Organization**: Group related functionality under a common interface

### Basic Implementation

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate and return the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate and return the perimeter of the shape"""
        pass
    
    # Concrete method (can be inherited as-is)
    def describe(self):
        return f"This is a {self.__class__.__name__} with area {self.area():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Usage
shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print(shape.describe())
    print(f"Perimeter: {shape.perimeter():.2f}\n")
```

### Advanced Example: Database Connection Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json

class DatabaseConnection(ABC):
    """Abstract base class for database connections"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close the database connection"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        """Execute INSERT, UPDATE, DELETE commands"""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> None:
        """Start a database transaction"""
        pass
    
    @abstractmethod
    def commit_transaction(self) -> bool:
        """Commit the current transaction"""
        pass
    
    @abstractmethod
    def rollback_transaction(self) -> bool:
        """Rollback the current transaction"""
        pass
    
    # Concrete method available to all subclasses
    def execute_batch(self, commands: List[tuple]) -> List[bool]:
        """Execute multiple commands in a transaction"""
        self.begin_transaction()
        results = []
        try:
            for command, params in commands:
                result = self.execute_command(command, params)
                results.append(result)
            self.commit_transaction()
            return results
        except Exception as e:
            self.rollback_transaction()
            print(f"Batch execution failed: {e}")
            return [False] * len(commands)

class SQLiteConnection(DatabaseConnection):
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.connection = None
        self.in_transaction = False
    
    def connect(self) -> bool:
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            return True
        except Exception as e:
            print(f"SQLite connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            return True
        except Exception as e:
            print(f"SQLite disconnection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        if not self.connection:
            raise RuntimeError("No active connection")
        
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        if not self.connection:
            raise RuntimeError("No active connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(command, params or ())
            if not self.in_transaction:
                self.connection.commit()
            return True
        except Exception as e:
            print(f"Command execution failed: {e}")
            return False
    
    def begin_transaction(self) -> None:
        self.in_transaction = True
    
    def commit_transaction(self) -> bool:
        try:
            self.connection.commit()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"Transaction commit failed: {e}")
            return False
    
    def rollback_transaction(self) -> bool:
        try:
            self.connection.rollback()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"Transaction rollback failed: {e}")
            return False

class MockDatabaseConnection(DatabaseConnection):
    """Mock implementation for testing purposes"""
    
    def __init__(self):
        self.connected = False
        self.data = {}
        self.in_transaction = False
        self.transaction_data = None
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        self.connected = False
        return True
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        if not self.connected:
            raise RuntimeError("No active connection")
        # Simulate query results
        return [{"id": 1, "name": "Test"}, {"id": 2, "name": "Sample"}]
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        if not self.connected:
            raise RuntimeError("No active connection")
        print(f"Mock executing: {command} with params: {params}")
        return True
    
    def begin_transaction(self) -> None:
        self.in_transaction = True
        self.transaction_data = self.data.copy()
    
    def commit_transaction(self) -> bool:
        self.in_transaction = False
        self.transaction_data = None
        return True
    
    def rollback_transaction(self) -> bool:
        if self.transaction_data is not None:
            self.data = self.transaction_data
        self.in_transaction = False
        self.transaction_data = None
        return True

# Polymorphic database manager
class DatabaseManager:
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
    
    def setup_database(self) -> bool:
        """Setup database tables - works with any connection type"""
        if not self.connection.connect():
            return False
        
        commands = [
            ("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)", ()),
            ("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)", ())
        ]
        
        results = self.connection.execute_batch(commands)
        return all(results)
    
    def add_user(self, name: str, email: str) -> bool:
        return self.connection.execute_command(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
    
    def get_users(self) -> List[Dict[str, Any]]:
        return self.connection.execute_query("SELECT * FROM users")

# Usage with different connection types
sqlite_db = DatabaseManager(SQLiteConnection("test.db"))
mock_db = DatabaseManager(MockDatabaseConnection())

# Both work identically due to polymorphism
for db_manager in [sqlite_db, mock_db]:
    db_manager.setup_database()
    db_manager.add_user("John Doe", "john@example.com")
    users = db_manager.get_users()
    print(f"Users: {users}")
```

---

## 6. Empty Classes

### Purpose and Use Cases

Empty classes (classes with only the `pass` statement) serve as lightweight containers for data or as placeholders during development. They're particularly useful for creating simple namespace objects or configuration containers.

### Common Applications

- **Configuration Objects**: Store settings and parameters
- **Data Transfer Objects**: Pass data between functions/modules
- **Namespace Objects**: Group related attributes
- **Rapid Prototyping**: Create placeholder classes during development

### Basic Examples

```python
# Simple configuration class
class DatabaseConfig:
    pass

# Create and configure
db_config = DatabaseConfig()
db_config.host = "localhost"
db_config.port = 5432
db_config.database = "myapp"
db_config.username = "admin"
db_config.password = "secret"

print(f"Connecting to {db_config.host}:{db_config.port}")
```

### Advanced Example: Game Object System

```python
class GameObject:
    pass

class Transform:
    pass

class Renderer:
    pass

# Create a game object with components
player = GameObject()
player.name = "Player Character"
player.health = 100
player.level = 1

# Add transform component
player.transform = Transform()
player.transform.position = (0, 0, 0)
player.transform.rotation = (0, 0, 0)
player.transform.scale = (1, 1, 1)

# Add renderer component
player.renderer = Renderer()
player.renderer.sprite = "player_sprite.png"
player.renderer.visible = True

# Game logic functions
def move_object(game_object, delta_x, delta_y, delta_z):
    transform = game_object.transform
    transform.position = (
        transform.position[0] + delta_x,
        transform.position[1] + delta_y,
        transform.position[2] + delta_z
    )

def render_object(game_object):
    if hasattr(game_object, 'renderer') and game_object.renderer.visible:
        print(f"Rendering {game_object.name} at {game_object.transform.position}")

# Usage
    
    def __post_init__(self):
        self._validate_username()
        self._validate_email()
        self._validate_age()
    
    def _validate_username(self):
        if not self.username or len(self.username) < 3:
            raise ValueError("Username must be at least 3 characters long")
        if not self.username.replace('_', '').replace('-', '').isalnum():
            raise ValueError("Username can only contain letters, numbers, hyphens, and underscores")
    
    def _validate_email(self):
        if not re.match(self._email_pattern, self.email):
            raise ValueError("Invalid email format")
    
    def _validate_age(self):
        if not 0 <= self.age <= 150:
            raise ValueError("Age must be between 0 and 150")

# Usage
try:
    user = User("john_doe", "john@example.com", 25)
    print(f"Valid user: {user}")
    
    invalid_user = User("jo", "invalid-email", -5)  # This will raise ValueError
except ValueError as e:
    print(f"Validation error: {e}")
```

---

## 8. Keyword Arguments

### Fundamentals and Best Practices

Keyword arguments allow you to pass arguments to functions by specifying the parameter name, making function calls more readable and flexible. They're essential for creating maintainable APIs and handling optional parameters.

### Types of Parameters

1. **Positional Arguments**: Must be provided in order
2. **Keyword Arguments**: Can be provided in any order using `name=value`
3. **Default Arguments**: Have default values, making them optional
4. **Variable Arguments**: `*args` and `**kwargs` for flexible parameter lists

### Comprehensive Example

```python
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta

def create_user_account(
    username: str,                    # Required positional argument
    email: str,                      # Required positional argument
    *,                               # Force keyword-only arguments after this
    password: Optional[str] = None,   # Keyword-only with default
    full_name: str = "",             # Keyword-only with default
    age: Optional[int] = None,       # Keyword-only with default
    preferences: Optional[Dict[str, Any]] = None,  # Keyword-only with default
    notification_settings: Optional[Dict[str, bool]] = None,  # Keyword-only
    auto_verify: bool = False,       # Keyword-only with default
    account_type: str = "standard",  # Keyword-only with default
    **additional_metadata           # Collect any extra keyword arguments
) -> Dict[str, Any]:
    """
    Create a user account with comprehensive configuration options.
    
    Args:
        username: Unique username for the account
        email: User's email address
        password: Account password (if None, temporary password is generated)
        full_name: User's full name
        age: User's age
        preferences: Dictionary of user preferences
        notification_settings: Dictionary of notification preferences
        auto_verify: Whether to automatically verify the account
        account_type: Type of account (standard, premium, admin)
        **additional_metadata: Any additional metadata to store
    
    Returns:
        Dictionary containing account information
    """
    
    # Set defaults for mutable arguments
    if preferences is None:
        preferences = {
            "theme": "light",
            "language": "en",
            "timezone": "UTC"
        }
    
    if notification_settings is None:
        notification_settings = {
            "email_notifications": True,
            "push_notifications": False,
            "sms_notifications": False
        }
    
    # Generate temporary password if not provided
    if password is None:
        import secrets
        password = f"temp_{secrets.token_urlsafe(8)}"
    
    # Create account data
    account = {
        "username": username,
        "email": email,
        "password_hash": hash(password),  # In reality, use proper password hashing
        "full_name": full_name,
        "age": age,
        "preferences": preferences,
        "notification_settings": notification_settings,
        "account_type": account_type,
        "created_at": datetime.now().isoformat(),
        "verified": auto_verify,
        "metadata": additional_metadata
    }
    
    # Account validation
    if age is not None and (age < 13 or age > 120):
        raise ValueError("Age must be between 13 and 120")
    
    if account_type not in ["standard", "premium", "admin"]:
        raise ValueError("Invalid account type")
    
    return account

# Demonstration of different calling patterns
print("=== Different ways to call the function ===\n")

# 1. Minimal call with required arguments only
basic_account = create_user_account("john_doe", "john@example.com")
print("Basic account:")
print(f"Username: {basic_account['username']}")
print(f"Verified: {basic_account['verified']}")
print()

# 2. Call with some optional keyword arguments
enhanced_account = create_user_account(
    "alice_smith", 
    "alice@example.com",
    full_name="Alice Smith",
    age=28,
    auto_verify=True
)
print("Enhanced account:")
print(f"Full name: {enhanced_account['full_name']}")
print(f"Age: {enhanced_account['age']}")
print(f"Verified: {enhanced_account['verified']}")
print()

# 3. Call with custom preferences and notification settings
custom_account = create_user_account(
    "bob_wilson",
    "bob@example.com",
    password="secure123",
    preferences={
        "theme": "dark",
        "language": "es",
        "timezone": "America/New_York"
    },
    notification_settings={
        "email_notifications": False,
        "push_notifications": True,
        "sms_notifications": True
    },
    account_type="premium"
)
print("Custom account:")
print(f"Theme: {custom_account['preferences']['theme']}")
print(f"Account type: {custom_account['account_type']}")
print()

# 4. Call with additional metadata
admin_account = create_user_account(
    "admin_user",
    "admin@example.com",
    full_name="System Administrator",
    account_type="admin",
    auto_verify=True,
    # Additional metadata via **kwargs
    department="IT",
    employee_id="EMP001",
    clearance_level=5,
    last_login=datetime.now().isoformat()
)
print("Admin account with metadata:")
print(f"Department: {admin_account['metadata']['department']}")
print(f"Clearance level: {admin_account['metadata']['clearance_level']}")
```

### Function Overloading Pattern

```python
def send_message(
    recipient: str,
    message: str,
    *,
    sender: str = "System",
    priority: str = "normal",
    delivery_method: str = "email",
    schedule_time: Optional[datetime] = None,
    retry_attempts: int = 3,
    include_attachments: bool = False,
    **options
):
    """Flexible message sending function"""
    
    message_data = {
        "to": recipient,
        "from": sender,
        "content": message,
        "priority": priority,
        "method": delivery_method,
        "retry_attempts": retry_attempts,
        "scheduled": schedule_time.isoformat() if schedule_time else None,
        "attachments": include_attachments,
        "options": options,
        "timestamp": datetime.now().isoformat()
    }
    
    print(f"Sending {priority} priority {delivery_method} message:")
    print(f"To: {recipient}, From: {sender}")
    print(f"Content: {message}")
    if schedule_time:
        print(f"Scheduled for: {schedule_time}")
    if options:
        print(f"Additional options: {options}")
    print("-" * 40)
    
    return message_data

# Different usage patterns
# Simple message
send_message("user@example.com", "Welcome to our service!")

# Urgent email
send_message(
    "admin@example.com", 
    "Server alert detected",
    priority="urgent",
    sender="Monitoring System"
)

# Scheduled SMS
send_message(
    "+1234567890",
    "Appointment reminder",
    delivery_method="sms",
    schedule_time=datetime.now() + timedelta(hours=24),
    include_attachments=False
)

# Message with custom options
send_message(
    "customer@example.com",
    "Your order has shipped",
    delivery_method="email",
    include_attachments=True,
    # Custom options
    template="shipping_notification",
    tracking_number="TRK123456",
    estimated_delivery="2024-01-15"
)
```

### Class Method with Keyword Arguments

```python
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime

@dataclass
class Report:
    title: str
    content: str
    created_by: str
    created_at: datetime
    tags: List[str]
    metadata: Dict[str, Any]
    
    @classmethod
    def create_report(
        cls,
        title: str,
        content: str,
        *,
        author: str = "Anonymous",
        tags: Optional[List[str]] = None,
        report_type: str = "general",
        include_timestamp: bool = True,
        auto_tag: bool = True,
        **additional_data
    ) -> 'Report':
        """Factory method to create reports with intelligent defaults"""
        
        # Handle default values for mutable types
        if tags is None:
            tags = []
        
        # Auto-generate tags based on content and type
        if auto_tag:
            if "error" in content.lower() or "exception" in content.lower():
                tags.append("error-report")
            if "performance" in content.lower():
                tags.append("performance")
            if report_type != "general":
                tags.append(f"{report_type}-report")
        
        # Build metadata
        metadata = {
            "report_type": report_type,
            "auto_generated_tags": auto_tag,
            **additional_data
        }
        
        # Add timestamp to title if requested
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            title = f"[{timestamp}] {title}"
        
        return cls(
            title=title,
            content=content,
            created_by=author,
            created_at=datetime.now(),
            tags=tags,
            metadata=metadata
        )

# Usage examples
reports = [
    # Basic report
    Report.create_report("Daily Summary", "All systems operational"),
    
    # Error report with auto-tagging
    Report.create_report(
        "System Error Analysis",
        "Multiple exceptions occurred in the payment module",
        author="DevOps Team",
        report_type="incident",
        severity="high"
    ),
    
    # Performance report with custom metadata
    Report.create_report(
        "Q4 Performance Review",
        "Performance metrics show 15% improvement",
        author="Analytics Team",
        tags=["quarterly", "metrics"],
        report_type="performance",
        quarter="Q4",
        year=2024,
        improvement_percentage=15.0
    )
]

for report in reports:
    print(f"Title: {report.title}")
    print(f"Tags: {report.tags}")
    print(f"Metadata: {report.metadata}")
    print("-" * 50)
```

---

## Additional Resources

### Official Documentation
- [Python Official OOP Tutorial](https://docs.python.org/3/tutorial/classes.html)
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html)
- [Abstract Base Classes](https://docs.python.org/3/library/abc.html)
- [Dataclasses Documentation](https://docs.python.org/3/library/dataclasses.html)

### Interactive Learning Platforms
- [Real Python - OOP Tutorials](https://realpython.com/python3-object-oriented-programming/)
- [Python.org Tutorial](https://docs.python.org/3/tutorial/)
- [Codecademy Python Course](https://www.codecademy.com/learn/learn-python-3)
- [Python Principles](https://pythonprinciples.com/)

### Practice Platforms
- [LeetCode OOP Problems](https://leetcode.com/problemset/all/)
- [HackerRank Python Domain](https://www.hackerrank.com/domains/python)
- [Codewars Python Kata](https://www.codewars.com/kata/search/python)
- [Exercism Python Track](https://exercism.org/tracks/python)

### Advanced Learning Resources
- [Effective Python by Brett Slatkin](https://effectivepython.com/)
- [Architecture Patterns with Python](https://www.cosmicpython.com/)
- [Python Tricks: The Book](https://realpython.com/products/python-tricks-book/)
- [Design Patterns in Python](https://python-patterns.guide/)

### Video Tutorials
- [Python OOP Tutorials - Tech With Tim](https://www.youtube.com/playlist?list=PLzMcBGfZo4-l1MqB1zoYfqzlj_HH-ZzXt)
- [Object Oriented Programming - Socratica](https://www.youtube.com/watch?v=apACNr7DC_s)
- [Python OOP - Programming with Mosh](https://www.youtube.com/watch?v=MikphENIrOo)

---

## Practice Exercises

### Exercise 1: E-commerce System
Create a comprehensive e-commerce system with the following requirements:
- Abstract `Product` class with concrete implementations for `Book`, `Electronics`, and `Clothing`
- Shopping cart with operator overloading for adding/removing items
- Order processing system using polymorphism
- Customer class using data classes

### Exercise 2: Game Development Framework
Design a simple game framework featuring:
- Abstract `GameObject` class
- Concrete classes for `Player`, `Enemy`, and `PowerUp`
- Collision detection system using magic methods
- Game state management with empty classes for configuration

### Exercise 3: API Client Library
Build a flexible API client library with:
- Abstract `APIClient` base class
- Implementations for different API providers (REST, GraphQL, etc.)
- Request/Response classes using data classes
- Authentication system with keyword arguments

### Exercise 4: Task Management System
Create a task management system incorporating:
- Abstract task types with polymorphic execution
- Priority queue using operator overloading
- Configuration management with empty classes
- Report generation with flexible keyword arguments

### Assignment Solutions

You can find complete solutions to all assignments and exercises in the `/solutions` directory of this repository. Each solution includes:
- Complete implementation
- Comprehensive comments
- Unit tests
- Usage examples

---

## Contributing

We welcome contributions to improve this tutorial! Please feel free to:
- Submit bug reports and feature requests
- Improve documentation and examples
- Add new exercises and solutions
- Share your learning experiences

---

## License

This tutorial is released under the MIT License. Feel free to use it for educational purposes and share it with others learning Python OOP concepts.

---

*Last updated: September 22, 2025*
animals = [Dog(), Cat(), Bird()]
for animal in animals:
    print(make_animal_speak(animal))
```

### Advanced Example: AI Model Interface

```python
class AIModel:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def predict(self, input_data):
        raise NotImplementedError("Subclasses must implement predict method")

class ImageClassifier(AIModel):
    def predict(self, image_data):
        # Simulate image classification
        print(f"{self.model_name}: Analyzing image...")
        return "Object detected: Cat"

class TextAnalyzer(AIModel):
    def predict(self, text_data):
        # Simulate text analysis
        print(f"{self.model_name}: Analyzing text sentiment...")
        return "Sentiment: Positive"

class VoiceRecognizer(AIModel):
    def predict(self, audio_data):
        # Simulate voice recognition
        print(f"{self.model_name}: Processing audio...")
        return "Transcription: Hello World"

# Polymorphic usage
def process_data(model, data):
    return model.predict(data)

models = [
    ImageClassifier("ResNet-50"),
    TextAnalyzer("BERT"),
    VoiceRecognizer("Whisper")
]

sample_data = ["image.jpg", "Great product!", "audio.wav"]

for i, model in enumerate(models):
    result = process_data(model, sample_data[i])
    print(f"Result: {result}\n")
```

### Assignment: Vehicle Management System

Create a polymorphic vehicle management system:

```python
# Your task: Implement the following classes
class Vehicle:
    def start_engine(self):
        pass
    
    def get_info(self):
        pass

class Car(Vehicle):
    # Implement start_engine and get_info methods
    pass

class Motorcycle(Vehicle):
    # Implement start_engine and get_info methods
    pass

class Bicycle(Vehicle):
    # Implement start_engine and get_info methods
    pass
```

---

## 2. Operator Overloading

### Definition and Purpose

Operator overloading is a specific form of polymorphism that allows you to define custom behaviors for built-in operators (`+`, `-`, `*`, `==`, etc.) when used with your custom objects. This is achieved through special "dunder" (double underscore) methods.

### Common Operator Methods

| Operator | Method | Description |
|----------|---------|-------------|
| `+` | `__add__(self, other)` | Addition |
| `-` | `__sub__(self, other)` | Subtraction |
| `*` | `__mul__(self, other)` | Multiplication |
| `/` | `__truediv__(self, other)` | Division |
| `==` | `__eq__(self, other)` | Equality |
| `<` | `__lt__(self, other)` | Less than |
| `>` | `__gt__(self, other)` | Greater than |
| `len()` | `__len__(self)` | Length |

### Basic Example: Money Class

```python
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot add {self.currency} and {other.currency}")
            return Money(self.amount + other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount + other, self.currency)
        return NotImplemented
    
    def __sub__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot subtract {other.currency} from {self.currency}")
            return Money(self.amount - other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount - other, self.currency)
        return NotImplemented
    
    def __mul__(self, multiplier):
        if isinstance(multiplier, (int, float)):
            return Money(self.amount * multiplier, self.currency)
        return NotImplemented
    
    def __eq__(self, other):
        if isinstance(other, Money):
            return self.amount == other.amount and self.currency == other.currency
        return False
    
    def __lt__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError(f"Cannot compare {self.currency} and {other.currency}")
            return self.amount < other.amount
        return NotImplemented
    
    def __repr__(self):
        return f"Money({self.amount}, '{self.currency}')"
    
    def __str__(self):
        return f"{self.amount} {self.currency}"

# Usage examples
wallet1 = Money(100, "USD")
wallet2 = Money(50, "USD")
cash = Money(25, "USD")

total = wallet1 + wallet2 + cash
print(total)  # 175 USD

doubled = wallet1 * 2
print(doubled)  # 200 USD

print(wallet1 > wallet2)  # True
```

### Advanced Example: Vector Class

```python
import math

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            return Vector(self.x * scalar, self.y * scalar)
        return NotImplemented
    
    def __truediv__(self, scalar):
        if isinstance(scalar, (int, float)) and scalar != 0:
            return Vector(self.x / scalar, self.y / scalar)
        return NotImplemented
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-10 and abs(self.y - other.y) < 1e-10
    
    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def __str__(self):
        return f"Vector({self.x:.2f}, {self.y:.2f})"
    
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    # Dot product using @ operator
    def __matmul__(self, other):
        return self.x * other.x + self.y * other.y

# Usage
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1 + v2)      # Vector(4.00, 6.00)
print(v1 * 2)       # Vector(6.00, 8.00)
print(abs(v1))      # 5.0
print(v1 @ v2)      # 11 (dot product)
```

### Assignment: Complex Number Class

Create a comprehensive complex number class with operator overloading.

---

## 3. Magic (Dunder) Methods

### Overview

Magic methods (also called dunder methods) are special methods in Python that start and end with double underscores. They allow your objects to integrate seamlessly with Python's built-in functions and operators.

### Categories of Magic Methods

#### 1. Object Representation
- `__str__(self)`: Human-readable string representation
- `__repr__(self)`: Developer-friendly representation
- `__format__(self, format_spec)`: Custom formatting

#### 2. Numeric Operations
- `__add__`, `__sub__`, `__mul__`, etc.: Arithmetic operations
- `__iadd__`, `__isub__`, `__imul__`, etc.: In-place operations

#### 3. Comparison Operations
- `__eq__`, `__ne__`, `__lt__`, `__le__`, `__gt__`, `__ge__`: Comparisons

#### 4. Container Emulation
- `__len__(self)`: Length of container
- `__getitem__(self, key)`: Item access with `obj[key]`
- `__setitem__(self, key, value)`: Item assignment
- `__delitem__(self, key)`: Item deletion
- `__contains__(self, item)`: Membership testing with `in`
- `__iter__(self)`: Iterator protocol

#### 5. Callable Objects
- `__call__(self, *args, **kwargs)`: Make objects callable like functions

### Comprehensive Example: Smart Dictionary

```python
class SmartDict:
    def __init__(self, data=None):
        self._data = data or {}
        self._access_count = {}
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, key):
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found")
        self._access_count[key] = self._access_count.get(key, 0) + 1
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
        if key not in self._access_count:
            self._access_count[key] = 0
    
    def __delitem__(self, key):
        if key in self._data:
            del self._data[key]
            del self._access_count[key]
    
    def __contains__(self, key):
        return key in self._data
    
    def __iter__(self):
        return iter(self._data)
    
    def __str__(self):
        return str(self._data)
    
    def __repr__(self):
        return f"SmartDict({self._data})"
    
    def __call__(self, key):
        """Make the dictionary callable to get access count"""
        return self._access_count.get(key, 0)
    
    def most_accessed(self):
        if not self._access_count:
            return None
        return max(self._access_count.items(), key=lambda x: x[1])

# Usage
smart_dict = SmartDict({'a': 1, 'b': 2, 'c': 3})

print(len(smart_dict))    # 3
print(smart_dict['a'])    # 1
print('a' in smart_dict)  # True

smart_dict['d'] = 4
del smart_dict['b']

for key in smart_dict:
    print(f"{key}: {smart_dict[key]}")

print(smart_dict('a'))    # Access count for 'a'
print(smart_dict.most_accessed())  # Most accessed key
```

### File-like Object Example

```python
class MemoryFile:
    def __init__(self, initial_content=""):
        self._content = initial_content
        self._position = 0
        self._closed = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __len__(self):
        return len(self._content)
    
    def __str__(self):
        return self._content
    
    def write(self, text):
        if self._closed:
            raise ValueError("Cannot write to closed file")
        self._content += text
        return len(text)
    
    def read(self, size=-1):
        if self._closed:
            raise ValueError("Cannot read from closed file")
        if size == -1:
            result = self._content[self._position:]
            self._position = len(self._content)
        else:
            result = self._content[self._position:self._position + size]
            self._position += len(result)
        return result
    
    def close(self):
        self._closed = True

# Usage with context manager
with MemoryFile() as f:
    f.write("Hello, World!")
    print(len(f))  # 13
    print(str(f))  # Hello, World!
```

---

## 4. Dynamic Polymorphism

### Concept and Implementation

Dynamic polymorphism occurs when the specific method to be executed is determined at runtime based on the object's actual type. This is achieved through inheritance and method overriding, allowing you to write flexible code that works with entire families of objects.

### Benefits of Dynamic Polymorphism

- **Extensibility**: Easy to add new types without modifying existing code
- **Maintainability**: Central logic handles all subtypes uniformly
- **Code Reduction**: Eliminates complex conditional logic

### Real-World Example: Payment Processing System

```python
from abc import ABC, abstractmethod
from datetime import datetime
import uuid

class PaymentMethod(ABC):
    def __init__(self):
        self.transaction_id = str(uuid.uuid4())
        self.timestamp = datetime.now()
    
    @abstractmethod
    def process_payment(self, amount, currency="USD"):
        pass
    
    @abstractmethod
    def validate_payment_info(self):
        pass
    
    def get_transaction_details(self):
        return {
            'transaction_id': self.transaction_id,
            'timestamp': self.timestamp.isoformat(),
            'method': self.__class__.__name__
        }

class CreditCard(PaymentMethod):
    def __init__(self, card_number, cvv, expiry_date):
        super().__init__()
        self.card_number = self._mask_card_number(card_number)
        self.cvv = cvv
        self.expiry_date = expiry_date
    
    def _mask_card_number(self, card_number):
        return f"****-****-****-{card_number[-4:]}"
    
    def validate_payment_info(self):
        # Simulate validation logic
        return len(self.cvv) == 3 and self.expiry_date > datetime.now()
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid card information"}
        
        # Simulate payment processing
        print(f"Processing ${amount} {currency} via Credit Card {self.card_number}")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

class PayPal(PaymentMethod):
    def __init__(self, email, password):
        super().__init__()
        self.email = email
        self._password_hash = hash(password)  # In reality, use proper hashing
    
    def validate_payment_info(self):
        return "@" in self.email and "." in self.email
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid PayPal credentials"}
        
        print(f"Processing ${amount} {currency} via PayPal ({self.email})")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

class BankTransfer(PaymentMethod):
    def __init__(self, account_number, routing_number, bank_name):
        super().__init__()
        self.account_number = f"****{account_number[-4:]}"
        self.routing_number = routing_number
        self.bank_name = bank_name
    
    def validate_payment_info(self):
        return len(self.routing_number) == 9
    
    def process_payment(self, amount, currency="USD"):
        if not self.validate_payment_info():
            return {"status": "failed", "message": "Invalid bank information"}
        
        print(f"Processing ${amount} {currency} via Bank Transfer ({self.bank_name})")
        return {
            "status": "success",
            "amount": amount,
            "currency": currency,
            **self.get_transaction_details()
        }

# Payment processor that works with any payment method
class PaymentProcessor:
    def __init__(self):
        self.transaction_history = []
    
    def process_transaction(self, payment_method: PaymentMethod, amount: float):
        """This method works with any payment method - that's polymorphism!"""
        result = payment_method.process_payment(amount)
        self.transaction_history.append(result)
        return result
    
    def get_successful_transactions(self):
        return [t for t in self.transaction_history if t.get("status") == "success"]

# Usage demonstration
processor = PaymentProcessor()

# Different payment methods
payment_methods = [
    CreditCard("1234567890123456", "123", datetime(2025, 12, 31)),
    PayPal("user@example.com", "password123"),
    BankTransfer("1234567890", "123456789", "Example Bank")
]

# Process payments polymorphically
amounts = [99.99, 149.50, 299.99]

for i, method in enumerate(payment_methods):
    result = processor.process_transaction(method, amounts[i])
    print(f"Transaction result: {result['status']}")
    print("-" * 50)

print(f"Successful transactions: {len(processor.get_successful_transactions())}")
```

### Media Player Example

```python
class MediaPlayer(ABC):
    @abstractmethod
    def play(self):
        pass
    
    @abstractmethod
    def pause(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass

class AudioPlayer(MediaPlayer):
    def __init__(self, audio_file):
        self.audio_file = audio_file
        self.is_playing = False
    
    def play(self):
        self.is_playing = True
        print(f"Playing audio: {self.audio_file}")
    
    def pause(self):
        self.is_playing = False
        print(f"Paused audio: {self.audio_file}")
    
    def stop(self):
        self.is_playing = False
        print(f"Stopped audio: {self.audio_file}")

class VideoPlayer(MediaPlayer):
    def __init__(self, video_file):
        self.video_file = video_file
        self.is_playing = False
    
    def play(self):
        self.is_playing = True
        print(f"Playing video: {self.video_file}")
    
    def pause(self):
        self.is_playing = False
        print(f"Paused video: {self.video_file}")
    
    def stop(self):
        self.is_playing = False
        print(f"Stopped video: {self.video_file}")

# Polymorphic media controller
def control_media(player: MediaPlayer, action: str):
    if action == "play":
        player.play()
    elif action == "pause":
        player.pause()
    elif action == "stop":
        player.stop()

# Usage
players = [
    AudioPlayer("song.mp3"),
    VideoPlayer("movie.mp4")
]

for player in players:
    control_media(player, "play")
    control_media(player, "pause")
```

---

## 5. Abstract Classes and Methods

### Purpose and Benefits

Abstract classes serve as blueprints that cannot be instantiated directly. They define a contract that subclasses must follow by implementing all abstract methods. This ensures consistency across related classes and helps prevent errors during development.

### Key Features

- **Contract Enforcement**: Subclasses must implement all abstract methods
- **Documentation**: Abstract classes document the expected interface
- **Polymorphism Support**: Enable polymorphic behavior across implementations
- **Code Organization**: Group related functionality under a common interface

### Basic Implementation

```python
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate and return the area of the shape"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate and return the perimeter of the shape"""
        pass
    
    # Concrete method (can be inherited as-is)
    def describe(self):
        return f"This is a {self.__class__.__name__} with area {self.area():.2f}"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Usage
shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print(shape.describe())
    print(f"Perimeter: {shape.perimeter():.2f}\n")
```

### Advanced Example: Database Connection Interface

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import sqlite3
import json

class DatabaseConnection(ABC):
    """Abstract base class for database connections"""
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the database"""
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """Close the database connection"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results"""
        pass
    
    @abstractmethod
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        """Execute INSERT, UPDATE, DELETE commands"""
        pass
    
    @abstractmethod
    def begin_transaction(self) -> None:
        """Start a database transaction"""
        pass
    
    @abstractmethod
    def commit_transaction(self) -> bool:
        """Commit the current transaction"""
        pass
    
    @abstractmethod
    def rollback_transaction(self) -> bool:
        """Rollback the current transaction"""
        pass
    
    # Concrete method available to all subclasses
    def execute_batch(self, commands: List[tuple]) -> List[bool]:
        """Execute multiple commands in a transaction"""
        self.begin_transaction()
        results = []
        try:
            for command, params in commands:
                result = self.execute_command(command, params)
                results.append(result)
            self.commit_transaction()
            return results
        except Exception as e:
            self.rollback_transaction()
            print(f"Batch execution failed: {e}")
            return [False] * len(commands)

class SQLiteConnection(DatabaseConnection):
    def __init__(self, database_path: str):
        self.database_path = database_path
        self.connection = None
        self.in_transaction = False
    
    def connect(self) -> bool:
        try:
            self.connection = sqlite3.connect(self.database_path)
            self.connection.row_factory = sqlite3.Row  # Enable dict-like access
            return True
        except Exception as e:
            print(f"SQLite connection failed: {e}")
            return False
    
    def disconnect(self) -> bool:
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
            return True
        except Exception as e:
            print(f"SQLite disconnection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        if not self.connection:
            raise RuntimeError("No active connection")
        
        cursor = self.connection.cursor()
        cursor.execute(query, params or ())
        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        if not self.connection:
            raise RuntimeError("No active connection")
        
        try:
            cursor = self.connection.cursor()
            cursor.execute(command, params or ())
            if not self.in_transaction:
                self.connection.commit()
            return True
        except Exception as e:
            print(f"Command execution failed: {e}")
            return False
    
    def begin_transaction(self) -> None:
        self.in_transaction = True
    
    def commit_transaction(self) -> bool:
        try:
            self.connection.commit()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"Transaction commit failed: {e}")
            return False
    
    def rollback_transaction(self) -> bool:
        try:
            self.connection.rollback()
            self.in_transaction = False
            return True
        except Exception as e:
            print(f"Transaction rollback failed: {e}")
            return False

class MockDatabaseConnection(DatabaseConnection):
    """Mock implementation for testing purposes"""
    
    def __init__(self):
        self.connected = False
        self.data = {}
        self.in_transaction = False
        self.transaction_data = None
    
    def connect(self) -> bool:
        self.connected = True
        return True
    
    def disconnect(self) -> bool:
        self.connected = False
        return True
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        if not self.connected:
            raise RuntimeError("No active connection")
        # Simulate query results
        return [{"id": 1, "name": "Test"}, {"id": 2, "name": "Sample"}]
    
    def execute_command(self, command: str, params: Optional[tuple] = None) -> bool:
        if not self.connected:
            raise RuntimeError("No active connection")
        print(f"Mock executing: {command} with params: {params}")
        return True
    
    def begin_transaction(self) -> None:
        self.in_transaction = True
        self.transaction_data = self.data.copy()
    
    def commit_transaction(self) -> bool:
        self.in_transaction = False
        self.transaction_data = None
        return True
    
    def rollback_transaction(self) -> bool:
        if self.transaction_data is not None:
            self.data = self.transaction_data
        self.in_transaction = False
        self.transaction_data = None
        return True

# Polymorphic database manager
class DatabaseManager:
    def __init__(self, connection: DatabaseConnection):
        self.connection = connection
    
    def setup_database(self) -> bool:
        """Setup database tables - works with any connection type"""
        if not self.connection.connect():
            return False
        
        commands = [
            ("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT, email TEXT)", ()),
            ("CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user_id INTEGER, amount REAL)", ())
        ]
        
        results = self.connection.execute_batch(commands)
        return all(results)
    
    def add_user(self, name: str, email: str) -> bool:
        return self.connection.execute_command(
            "INSERT INTO users (name, email) VALUES (?, ?)",
            (name, email)
        )
    
    def get_users(self) -> List[Dict[str, Any]]:
        return self.connection.execute_query("SELECT * FROM users")

# Usage with different connection types
sqlite_db = DatabaseManager(SQLiteConnection("test.db"))
mock_db = DatabaseManager(MockDatabaseConnection())

# Both work identically due to polymorphism
for db_manager in [sqlite_db, mock_db]:
    db_manager.setup_database()
    db_manager.add_user("John Doe", "john@example.com")
    users = db_manager.get_users()
    print(f"Users: {users}")
```

---

## 6. Empty Classes

### Purpose and Use Cases

Empty classes (classes with only the `pass` statement) serve as lightweight containers for data or as placeholders during development. They're particularly useful for creating simple namespace objects or configuration containers.

### Common Applications

- **Configuration Objects**: Store settings and parameters
- **Data Transfer Objects**: Pass data between functions/modules
- **Namespace Objects**: Group related attributes
- **Rapid Prototyping**: Create placeholder classes during development

### Basic Examples

```python
# Simple configuration class
class DatabaseConfig:
    pass

# Create and configure
db_config = DatabaseConfig()
db_config.host = "localhost"
db_config.port = 5432
db_config.database = "myapp"
db_config.username = "admin"
db_config.password = "secret"

print(f"Connecting to {db_config.host}:{db_config.port}")
```

### Advanced Example: Game Object System

```python
class GameObject:
    pass

class Transform:
    pass

class Renderer:
    pass

# Create a game object with components
player = GameObject()
player.name = "Player Character"
player.health = 100
player.level = 1

# Add transform component
player.transform = Transform()
player.transform.position = (0, 0, 0)
player.transform.rotation = (0, 0, 0)
player.transform.scale = (1, 1, 1)

# Add renderer component
player.renderer = Renderer()
player.renderer.sprite = "player_sprite.png"
player.renderer.visible = True

# Game logic functions
def move_object(game_object, delta_x, delta_y, delta_z):
    transform = game_object.transform
    transform.position = (
        transform.position[0] + delta_x,
        transform.position[1] + delta_y,
        transform.position[2] + delta_z
    )

def render_object(game_object):
    if hasattr(game_object, 'renderer') and game_object.renderer.visible:
        print(f"Rendering {game_object.name} at {game_object.transform.position}")

# Usage
