# tweekz

## Overview

This tool is designed as a platform for demonstrating and executing image processing operations. It features a graphical user interface for ease of use, supporting basic functionalities such as loading, displaying, modifying, and saving images. The application is developed in Python, emphasizing modularity and extensibility to facilitate future additions of more complex image processing capabilities.

## Features

- Load and display images from your file system.
- Save modified images back to your file system.
- Apply a zero intensity transformation to all pixels.
- Apply power law transformation
- Restore the original state of any modified image.

## Getting Started

### Prerequisites

- Python 3.8 or later

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/r-borgess/tweekz.git
   ```
2. Navigate to the project directory:
   ```
   cd tweekz
   ```
3. Create and activate a virtual environment:
   - **Windows:**
     ```
     python -m venv venv
     venv\Scripts\activate
     ```
   - **Unix/MacOS:**
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Application

Execute the main script:
```
python src/main.py
```

## How to Use

- Use the GUI to interact with the application's features: load, modify, save images, and reset modifications.

## Contributing

Contributions are welcome. Please feel free to contribute by opening a pull request or filing an issue.

## License

This project is available under the MIT License. See the LICENSE file for more details.
