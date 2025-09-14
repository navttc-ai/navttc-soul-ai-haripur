# Day 4: In-Depth Python Data Structures

Welcome! This repository contains educational materials for a deep dive into Python's fundamental data structures, based on the "Day 4" lecture. The goal is to provide a clear understanding of how Python handles variables and the core differences and use-cases for strings, lists, tuples, dictionaries, and sets.

## Core Concepts in Python

Before diving into specific data structures, it's crucial to understand two foundational concepts in Python.

### 1. Variables as References (Labels, Not Boxes)
In Python, a variable is not a container that holds data. Instead, it's a **label** or a **name** that points to an object stored in memory. When you write `x = 100`, you are creating an integer object with the value `100` and making the name `x` point to it. This is why assigning one variable to another (`y = x`) makes both names point to the *exact same object*.

### 2. Mutable vs. Immutable Data Types
This is a direct consequence of the reference model.

-   **Immutable Objects**: These cannot be changed after they are created. Any operation that appears to "modify" them actually creates a new object in memory.
    -   *Examples*: `integers`, `strings`, `tuples`.
-   **Mutable Objects**: These can be changed "in-place" without creating a new object. Modifying the object will affect all variables that reference it.
    -   *Examples*: `lists`, `dictionaries`, `sets`.

## Data Structures Overview

This lesson covers the following data structures in detail:

### Strings (`str`)
-   **Type**: Immutable
-   **Description**: An ordered sequence of characters.
-   **Key Methods**:
    -   `.capitalize()`: Returns a new string with the first character capitalized.
    -   `.replace(old, new)`: Returns a new string with occurrences of `old` replaced by `new`.
    -   `.find(substring)`: Returns the starting index of a substring.
    -   `.split()`: Splits the string into a list of words.
    -   `.upper()`: Returns a new string in all uppercase.

### Lists (`list`)
-   **Type**: Mutable
-   **Description**: An ordered, flexible collection of items. The workhorse of Python data structures.
-   **Key Methods**:
    -   `.append(item)`: Adds an item to the end of the list.
    -   `.insert(index, item)`: Inserts an item at a specific position.
    -   `.pop()`: Removes and returns the last item.
    -   `.remove(item)`: Removes the first occurrence of an item.
    -   `.sort()`: Sorts the list in-place.

### Tuples (`tuple`)
-   **Type**: Immutable
-   **Description**: An ordered, fixed-size collection of items. Often used for data that should not change, like database records or coordinates.
-   **Key Features**: Supports "packing" (creating a tuple) and "unpacking" (assigning its values to multiple variables).
-   **Key Methods**:
    -   `.count(item)`: Counts how many times an item appears.
    -   `.index(item)`: Finds the index of the first occurrence of an item.

### Dictionaries (`dict`)
-   **Type**: Mutable
-   **Description**: An unordered collection of `key:value` pairs. Keys must be unique and immutable. Optimized for fast data retrieval.
-   **Key Methods**:
    -   `.get(key, default)`: Safely retrieves a value by its key, returning a default value if the key doesn't exist.
    -   `.pop(key)`: Removes a key-value pair and returns the value.
    -   `.items()`: Returns a view of all key-value pairs, useful for looping.

### Sets (`set`)
-   **Type**: Mutable
-   **Description**: An unordered collection of **unique** items. Duplicates are automatically removed.
-   **Key Features**: Highly efficient for membership testing and mathematical set operations.
-   **Key Operations**:
    -   `union (|)`: Combines all items from two sets.
    -   `intersection (&)`: Finds items that exist in both sets.
    -   `difference (-)`: Finds items that are in one set but not the other.

## Repository Contents

-   `Day4_Data_Structures_Detailed.pdf`: The original lecture slides.
-   `Day4_Data_Structures.ipynb`: A comprehensive Jupyter Notebook containing all lecture notes, code examples, and practice exercises with solutions.
-   `generate_notebook.py`: A Python script to programmatically generate the `Day4_Data_Structures.ipynb` file.
-   `create_readme.py`: The script used to generate this README file.

## Getting Started

To explore the code and exercises interactively, you can use the Jupyter Notebook.

1.  **Prerequisites**: Make sure you have Python and Jupyter Notebook installed on your system.
    ```bash
    pip install notebook
    ```
2.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
3.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```
4.  **Open the Notebook**: In the Jupyter interface that opens in your browser, click on `Day4_Data_Structures.ipynb` to get started.

