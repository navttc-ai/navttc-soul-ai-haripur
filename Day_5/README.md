# Day 5: Conditional Execution
## Python Programming Fundamentals - Professional Tutorial

---

## Table of Contents
1. [Introduction to Conditionals](#introduction-to-conditionals)
2. [The `if` Statement](#the-if-statement)
3. [Multiple Conditions with Logical Operators](#multiple-conditions-with-logical-operators)
4. [The `elif` Statement](#the-elif-statement)
5. [The `else` Statement](#the-else-statement)
6. [Complete Conditional Chains](#complete-conditional-chains)
7. [Loop Control Statements](#loop-control-statements)
8. [Nested Conditionals](#nested-conditionals)
9. [The Ternary Operator](#the-ternary-operator)
10. [Best Practices and Common Mistakes](#best-practices-and-common-mistakes)
11. [Practice Assignments](#practice-assignments)
12. [Conclusion](#conclusion)

---

## Introduction to Conditionals

### What are Conditionals?

Conditional statements are the **decision-making brain** of your Python programs. They allow your code to execute different blocks of instructions based on whether certain conditions are true or false. Think of conditionals as the "if-then" logic that governs how your program responds to different situations.

### Why Use Conditionals?

Conditionals are essential for creating dynamic, interactive programs. Here are some real-world applications:

- **Authentication Systems**: Check passwords and user credentials
- **E-commerce Platforms**: Apply discounts and calculate pricing
- **Form Validation**: Ensure user input meets requirements
- **Game Development**: Create interactive gameplay mechanics
- **Data Processing**: Filter and categorize information

### Basic Structure

All conditional statements in Python follow this fundamental pattern:

```python
if condition:
    # Code to execute when condition is True
    do_something()
```

**Key Points:**
- The `condition` must evaluate to `True` or `False` (boolean value)
- The colon (`:`) is mandatory after the condition
- Proper indentation (typically 4 spaces) is required for the code block

---

## The `if` Statement

The `if` statement is the most basic conditional structure. It executes a block of code only when the specified condition evaluates to `True`.

### Syntax and Example

```python
# Basic if statement
age = 18
if age >= 18:
    print("You can vote!")
```

### How It Works

1. Python evaluates the condition `age >= 18`
2. If the condition is `True`, the indented code block executes
3. If the condition is `False`, the code block is skipped

### Practical Example

```python
# Password validation
password = "secure123"
if len(password) >= 8:
    print("Password meets minimum length requirement")
```

---

## Multiple Conditions with Logical Operators

You can combine multiple conditions using logical operators to create more complex decision-making logic.

### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `and` | Both conditions must be True | `age >= 18 and has_license` |
| `or` | At least one condition must be True | `is_weekend or is_holiday` |
| `not` | Inverts the boolean value | `not is_logged_in` |

### Example with `and` Operator

```python
# Driver eligibility check
age = 20
has_license = True

if age >= 18 and has_license:
    print("You can drive!")
```

### Example with `or` Operator

```python
# Access control
is_admin = False
is_owner = True

if is_admin or is_owner:
    print("Access granted to sensitive data")
```

---

## The `elif` Statement

The `elif` (else if) statement allows you to check multiple conditions in sequence. It's used when you have several mutually exclusive conditions to evaluate.

### Basic `elif` Example

```python
# Grade assignment
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
elif score >= 60:
    grade = "D"
else:
    grade = "F"

print(f"Grade: {grade}")
```

### How `elif` Works

1. Python evaluates conditions **from top to bottom**
2. Only the **first** `True` condition executes
3. All subsequent conditions are **skipped** once a match is found
4. This creates an **exclusive decision tree**

### Practical Example

```python
# Weather-based clothing recommendation
temperature = 22

if temperature > 30:
    print("Wear light clothes and stay hydrated")
elif temperature > 20:
    print("Perfect weather for a t-shirt")
elif temperature > 10:
    print("Consider wearing a light jacket")
else:
    print("Bundle up with warm clothes")
```

---

## The `else` Statement

The `else` statement provides a **fallback option** when all previous conditions are `False`. It acts as the "default case" in your conditional logic.

### Basic `else` Example

```python
# Temperature assessment
temperature = 15

if temperature > 25:
    print("It's hot!")
elif temperature > 15:
    print("It's warm")
else:
    print("It's cold!")
```

### When to Use `else`

- As a **catch-all** for unhandled cases
- To provide **default behavior**
- To ensure your program **always takes an action**

---

## Complete Conditional Chains

A complete conditional chain handles **every possible scenario** your program might encounter. This ensures robust and predictable behavior.

### Example: Number Classification

```python
# Complete chain handling all possibilities
num = 0

if num > 0:
    print("Positive")
elif num < 0:
    print("Negative")
else:
    print("Zero")
```

### Benefits of Complete Chains

- **Comprehensive coverage** of all scenarios
- **Predictable program behavior**
- **Easier debugging** and maintenance
- **Better user experience**

---

## Loop Control Statements

Loop control statements modify the normal flow of loops, giving you fine-grained control over iteration behavior.

### The `break` Statement

The `break` statement **immediately terminates** the current loop and continues execution after the loop.

```python
# Finding the first large number
numbers = [1, 3, 8, 12, 15]

for num in numbers:
    if num > 10:
        print(f"Found large number: {num}")
        break
    print(num)

# Output: 1, 3, 8, Found large number: 12
```

#### When to Use `break`

- **Search operations**: Stop when you find what you need
- **Menu systems**: Exit when user chooses 'quit'
- **Error handling**: Stop processing when an error occurs
- **Performance optimization**: Avoid unnecessary iterations

### The `continue` Statement

The `continue` statement **skips the rest** of the current iteration and jumps to the next iteration.

```python
# Skip processing number 3
for i in range(1, 6):
    if i == 3:
        continue  # Skip 3
    print(i)

# Output: 1, 2, 4, 5
```

#### Practical Example: Data Validation

```python
# Skip invalid scores while processing valid ones
scores = [85, -1, 92, 0, 78]

for score in scores:
    if score <= 0:
        continue  # Skip invalid scores
    print(f"Valid score: {score}")

# Output: Valid score: 85, Valid score: 92, Valid score: 78
```

### The `pass` Statement

The `pass` statement is a **null operation** - it does nothing when executed. It's used as a placeholder for future code.

```python
# Placeholder for future implementation
def future_function():
    # I'll write this later
    pass

# Conditional placeholder
user_input = "help"
if user_input == "help":
    pass  # Will add help functionality later
else:
    print("Unknown command")
```

#### Why Use `pass`?

- **Syntactic requirement**: Python needs code after colons
- **Development workflow**: Plan structure before implementation
- **Gradual development**: Build functionality incrementally

---

## Nested Conditionals

Nested conditionals involve placing conditional statements **inside other conditional statements**. This creates a hierarchical decision tree for complex logic.

### Simple Nested Example

```python
# Ticket pricing with multiple factors
age = 20
is_student = True

if age >= 18:
    if is_student:
        price = 10  # Student discount
    else:
        price = 15  # Regular adult price
else:
    price = 8  # Child price

print(f"Ticket: ${price}")
```

### Best Practices for Nesting

1. **Limit depth**: Maximum 2-3 levels to maintain readability
2. **Use clear names**: Variables should be self-explanatory
3. **Add comments**: Explain complex logic
4. **Consider alternatives**: Sometimes flattening is better

### Alternative to Deep Nesting

```python
# Instead of deep nesting, use logical operators
age = 20
is_student = True

if age < 18:
    price = 8
elif age >= 18 and is_student:
    price = 10
else:
    price = 15
```

---

## The Ternary Operator

The ternary operator provides a **concise way** to write simple if-else statements in a single line.

### Syntax

```python
value_if_true if condition else value_if_false
```

### Basic Example

```python
age = 20

# Traditional approach
if age >= 18:
    status = "Adult"
else:
    status = "Minor"

# Ternary operator (concise)
status = "Adult" if age >= 18 else "Minor"
print(status)  # Output: Adult
```

### More Ternary Examples

```python
# Even/odd check
number = 7
result = "Even" if number % 2 == 0 else "Odd"
print(f"Number {number} is: {result}")

# Membership discount
is_member = True
discount = 0.1 if is_member else 0.0
print(f"Discount: {discount * 100}%")

# Input validation
username = ""
name = username if username else "Guest"
print(f"Welcome, {name}!")
```

### When to Use Ternary Operators

- **Simple conditions**: Binary true/false scenarios
- **Variable assignment**: Setting values based on conditions
- **Inline logic**: When brevity improves readability
- **Avoid for complex logic**: Can reduce readability

---

## Best Practices and Common Mistakes

### Best Practices

#### 1. Use Meaningful Variable Names
```python
# Good
is_authenticated = check_user_credentials()
if is_authenticated:
    grant_access()

# Avoid
x = check_user_credentials()
if x:
    grant_access()
```

#### 2. Keep Conditions Simple and Readable
```python
# Good
if user.is_active and user.has_permission("read"):
    display_data()

# Avoid complex expressions
if user.status == "active" and user.permissions["read"] == True and user.account_type != "suspended":
    display_data()
```

#### 3. Test All Possible Paths
```python
# Ensure all scenarios are covered
def categorize_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"  # Don't forget the default case
```

#### 4. Add Comments for Complex Logic
```python
# Complex business logic should be documented
if (user.subscription_active and 
    user.payment_current and 
    not user.account_suspended):
    # User has full access to premium features
    enable_premium_features()
```

### Common Mistakes

#### 1. Missing Colon
```python
# Wrong
if age >= 18
    print("Adult")

# Correct
if age >= 18:
    print("Adult")
```

#### 2. Incorrect Indentation
```python
# Wrong
if age >= 18:
print("Adult")

# Correct
if age >= 18:
    print("Adult")
```

#### 3. Assignment vs. Comparison
```python
# Wrong (assignment)
if age = 18:
    print("Exactly 18")

# Correct (comparison)
if age == 18:
    print("Exactly 18")
```

#### 4. Not Handling All Cases
```python
# Incomplete - what if score is negative?
if score >= 60:
    print("Pass")
else:
    print("Fail")

# Better - handle invalid input
if score < 0 or score > 100:
    print("Invalid score")
elif score >= 60:
    print("Pass")
else:
    print("Fail")
```

---

## Practice Assignments

### Assignment 1: Number Classifier

**Objective**: Create a program that analyzes a user-provided number.

**Requirements**:
- Accept a number from the user
- Determine if it's positive, negative, or zero
- If the number is not zero, determine if it's even or odd
- Handle invalid input gracefully

**Solution Framework**:
```python
def number_classifier():
    try:
        num = float(input("Enter a number: "))
        
        # Classify sign
        if num > 0:
            print(f"{num} is positive")
        elif num < 0:
            print(f"{num} is negative")
        else:
            print(f"{num} is zero")
        
        # Check even/odd for integers
        if num != 0 and num.is_integer():
            if int(num) % 2 == 0:
                print(f"{int(num)} is even")
            else:
                print(f"{int(num)} is odd")
                
    except ValueError:
        print("Please enter a valid number!")
```

### Assignment 2: Grade Calculator

**Objective**: Build a comprehensive grade calculation system.

**Requirements**:
- Accept a test score (0-100)
- Assign appropriate letter grades
- Provide meaningful feedback messages
- Validate input range

**Grading Scale**:
- A: 90-100 (Excellent)
- B: 80-89 (Good)
- C: 70-79 (Satisfactory)
- D: 60-69 (Below Average)
- F: 0-59 (Failed)

**Solution Framework**:
```python
def grade_calculator():
    try:
        score = float(input("Enter test score (0-100): "))
        
        if score < 0 or score > 100:
            print("Score must be between 0 and 100!")
            return
        
        if score >= 90:
            grade = "A"
            message = "Excellent work! Outstanding performance!"
        elif score >= 80:
            grade = "B"
            message = "Good job! Above average performance!"
        elif score >= 70:
            grade = "C"
            message = "Satisfactory. You passed!"
        elif score >= 60:
            grade = "D"
            message = "Below average. Consider studying more."
        else:
            grade = "F"
            message = "Failed. Please retake the test."
        
        print(f"Score: {score}")
        print(f"Grade: {grade}")
        print(f"Message: {message}")
        
    except ValueError:
        print("Please enter a valid number!")
```

### Assignment 3: Simple Calculator

**Objective**: Develop a basic arithmetic calculator with error handling.

**Requirements**:
- Accept two numbers and an operation (+, -, *, /)
- Perform the calculation
- Handle division by zero
- Validate operations and input

**Solution Framework**:
```python
def simple_calculator():
    try:
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        operation = input("Enter operation (+, -, *, /): ").strip()
        
        if operation == "+":
            result = num1 + num2
            print(f"{num1} + {num2} = {result}")
        elif operation == "-":
            result = num1 - num2
            print(f"{num1} - {num2} = {result}")
        elif operation == "*":
            result = num1 * num2
            print(f"{num1} * {num2} = {result}")
        elif operation == "/":
            if num2 == 0:
                print("Error: Division by zero is not allowed!")
            else:
                result = num1 / num2
                print(f"{num1} / {num2} = {result}")
        else:
            print("Error: Invalid operation! Please use +, -, *, or /")
            
    except ValueError:
        print("Error: Please enter valid numbers!")
```

---

## Conclusion

### Key Takeaways

1. **Conditional statements** are fundamental for creating dynamic, responsive programs
2. **if, elif, and else** provide comprehensive decision-making capabilities
3. **Loop control statements** (break, continue, pass) offer fine-grained flow control
4. **Nested conditionals** handle complex scenarios but should be used judiciously
5. **Ternary operators** provide concise syntax for simple conditions
6. **Proper error handling** and input validation are essential for robust programs

### Skills Developed

- **Logical thinking**: Breaking down problems into conditional steps
- **Code organization**: Structuring decision trees effectively
- **Error handling**: Anticipating and managing edge cases
- **User experience**: Providing meaningful feedback and validation
- **Code quality**: Writing readable, maintainable conditional logic

### Next Steps

1. **Practice extensively**: Work through additional conditional logic problems
2. **Combine concepts**: Integrate conditionals with loops for more powerful programs
3. **Study boolean algebra**: Understand logical operators in depth
4. **Explore advanced patterns**: Learn about match-case statements (Python 3.10+)
5. **Build projects**: Apply conditional logic to real-world applications

### Resources for Further Learning

- **Official Python Documentation**: Detailed reference for control flow
- **Practice platforms**: LeetCode, HackerRank, Codewars for conditional problems
- **Project ideas**: Build a text-based adventure game, form validator, or decision tree classifier

---

*This tutorial provides a comprehensive foundation in Python conditional execution. Practice regularly and experiment with different scenarios to master these essential programming concepts.*
