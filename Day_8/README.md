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

```

###  Example

```python
# Base class (the general interface)
class AIModel:
    def load_model(self, path):
        # General logic to load a model file
        pass

    def predict(self, input_data):
        # This is a placeholder; it must be implemented by subclasses
        raise NotImplementedError("Subclasses must implement this method")

# Derived class 1: Image Classifier
class ImageClassifier(AIModel):
    def predict(self, image):
        # Specific implementation for processing an image
        print("Classifying image...")
        # ... image-specific logic ...
        return "Cat"

# Derived class 2: Text Analyzer
class TextAnalyzer(AIModel):
    def predict(self, text):
        # Specific implementation for processing text
        print("Analyzing text sentiment...")
        # ... text-specific logic ...
        return "Positive"

# --- Using Polymorphism ---
# We can treat both objects as if they are the same type (AIModel)
image_model = ImageClassifier()
text_model = TextAnalyzer()

models = [image_model, text_model]
input_data = ["some_image.jpg", "This movie was fantastic!"]

# The same function call `model.predict()` works for different model types
for i, model in enumerate(models):
    result = model.predict(input_data[i])
    print(f"Prediction: {result}\n")
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
# Operator Overloading: Example
class Money:
    def __init__(self, amount, currency):
        self.amount = amount
        self.currency = currency

    def __add__(self, other):
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)

    def __repr__(self):
        return f"{self.amount} {self.currency}"

wallet1 = Money(100, "USD")
wallet2 = Money(50, "USD")
print(wallet1 + wallet2)
```

### Advanced Example: Vector Class

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # This special method defines the behavior of the '+' operator
    def __add__(self, other_point):
        # Add the x and y coordinates separately
        new_x = self.x + other_point.x
        new_y = self.y + other_point.y
        return Point(new_x, new_y)

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# --- Using Operator Overloading ---
p1 = Point(1, 2)
p2 = Point(5, 3)

# Because we overloaded '+', this now works!
result = p1 + p2

print(result) # Output: Point(6, 5)
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

### Comprehensive Example: 

```python
# Dunder Representation: Example
class Book:
    def __init__(self, title, author):
        self.title, self.author = title, author

    def __str__(self): return f"{self.title} by {self.author}"
    def __repr__(self): return f"Book('{self.title}', '{self.author}')"

my_book = Book("Dune", "F. Herbert")
print(my_book)      # Calls __str__
print(repr(my_book)) # Calls __repr__

class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author


my_book = Book("Dune", "Frank Herbert")

print(my_book)       
print(repr(my_book))

class Book:
    def __init__(self, title, author):
        self.title = title
        self.author = author

    def __str__(self):
        # User-friendly output for print()
        return f"{self.title} by {self.author}"

    def __repr__(self):
        # Unambiguous representation for developers
        return f"Book(title='{self.title}', author='{self.author}')"

my_book = Book("Dune", "Frank Herbert")

print(my_book)        # Calls __str__ -> Output: Dune by Frank Herbert
print(repr(my_book))  # Calls __repr__ -> Output: Book(title='Dune', author='Frank Herbert')
```

### Another example

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # This special method defines the behavior of the '+' operator
    def __add__(self, other_point):
        # Add the x and y coordinates separately
        new_x = self.x + other_point.x
        new_y = self.y + other_point.y
        return Point(new_x, new_y)

    def __str__(self):
        return f"Point({self.x}, {self.y})"

# --- Using Operator Overloading ---
p1 = Point(1, 2)
p2 = Point(5, 3)

# Because we overloaded '+', this now works!
result = p1 + p2

print(result) # Output: Point(6, 5)
```

---

## 4. Dynamic Polymorphism

### Concept and Implementation

Dynamic polymorphism occurs when the specific method to be executed is determined at runtime based on the object's actual type. This is achieved through inheritance and method overriding, allowing you to write flexible code that works with entire families of objects.

### Benefits of Dynamic Polymorphism

- **Extensibility**: Easy to add new types without modifying existing code
- **Maintainability**: Central logic handles all subtypes uniformly
- **Code Reduction**: Eliminates complex conditional logic

### Real-World Example: 

```python
# Dynamic Polymorphism: Example
class Notification:
    def send(self, message): raise NotImplementedError

class Email(Notification):
    def send(self, message): print(f"Sending '{message}' via Email")

class SMS(Notification):
    def send(self, message): print(f"Sending '{message}' via SMS")

# This function works with any object that is a 'Notification'
def send_alert(notification_channel, message):
    notification_channel.send(message)

# The correct .send() method is called at runtime
send_alert(Email(), "Server is down!")
send_alert(SMS(), "Server is down!")
```

### Media Player Example

```python
# Base class
class Employee:
    def __init__(self, name):
        self.name = name

    def calculate_pay(self):
        # A generic placeholder implementation
        raise NotImplementedError("Subclasses must implement this method")
# Subclass 1
class SalariedEmployee(Employee):
    def __init__(self, name, weekly_salary):
        super().__init__(name)
        self.weekly_salary = weekly_salary

    def calculate_pay(self):
        return self.weekly_salary

# Subclass 2
class HourlyEmployee(Employee):
    def __init__(self, name, hours_worked, rate):
        super().__init__(name)
        self.hours_worked = hours_worked
        self.rate = rate

    def calculate_pay(self):
        return self.hours_worked * self.rate
# Base class
class Employee:
    def __init__(self, name):
        self.name = name

    def calculate_pay(self):
        # A generic placeholder implementation
        raise NotImplementedError("Subclasses must implement this method")
# Create instances of the subclasses
emp1 = SalariedEmployee("Alice", 1500)
emp2 = HourlyEmployee("Bob", 40, 50)
emp3 = HourlyEmployee("Charlie", 35, 45)

# Treat all objects as the base class type (Employee)
employees = [emp1, emp2, emp3]

# The same method call works differently for each object
for employee in employees:
    # Python dynamically checks the object's actual type at runtime
    # and calls the correct calculate_pay() method.
    pay = employee.calculate_pay()
    print(f"{employee.name}'s pay is: ${pay}")
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
# Abstract Class: Example
from abc import ABC, abstractmethod

class DataStorage(ABC):
    @abstractmethod
    def save(self, data):
        pass

    @abstractmethod
    def load(self):
        pass

class FileStorage(DataStorage):
    def save(self, data):
        print(f"Saving '{data}' to file...")

    def load(self):
        print("Loading from file...")
        return "some_data"

# you cannot create an instance of an abstract class
# storage = DataStorage() # This would raise a TypeError

# A concrete class can be instantiated
file_handler = FileStorage()
file_handler.save("My important data")
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
# Defining an empty class
class GameObject:
    pass

# Creating an instance
player = GameObject()

# Now we can add attributes to this empty object dynamically
player.name = "Arif"
player.score = 100
player.position = (10, 20)

print(f"Player: {player.name}, Score: {player.score}")
# Output: Player: Arif, Score: 100
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
# Data Class: Example
from dataclasses import dataclass

# Before (Manual)
class ManualPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __repr__(self):
        return f"ManualPoint(x={self.x}, y={self.y})"

# After (Automatic with dataclass)
@dataclass
class Point:
    x: int
    y: int

p1 = ManualPoint(10, 20)
p2 = Point(10, 20)

print(f"Manual Class: {p1}")
print(f"Data Class:   {p2}
```

### example

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
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"
        
    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

p1 = Point(1, 2)
print(p1) # Output: Point(x=1, y=2)
p2 = Point(1,2)
print(p1==p2)
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

# The __init__, __repr__, and __eq__ methods are all created automatically!
p1 = Point(1, 2)
p2 = Point(1, 2)

print(p1)         # Output: Point(x=1, y=2)
print(p1 == p2)   # Output: True
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
# Keyword Arguments: Example
def connect(host, port=5432, user="admin", ssl=False):
    print("--- Connection Details ---")
    print(f"Connecting to {host}:{port} as {user}...")
    print(f"SSL Enabled: {ssl}")

# Call using positional and default arguments
connect("db.example.com")

# Call overriding defaults using keywords, out of order
connect("db.example.com", ssl=True, user="guest")
def create_profile(name, role="user", is_active=True):
    print(f"Creating profile for {name}.")
    print(f"Role: {role}")
    print(f"Status: {'Active' if is_active else 'Inactive'}")
    print("-" * 20)

# --- Calling the function in different ways ---

# 1. Using only the required positional argument
create_profile("Zoya")

# 2. Overriding one default value using its keyword
create_profile("Bilal", role="admin")

# 3. Overriding all defaults, in a different order
create_profile("Fatima", is_active=False, role="moderator")
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
