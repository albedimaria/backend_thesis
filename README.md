# Essentia for Audio Processing in Python

This is a Essentia-based Python backend for processing audio files and providing analysis results. It allows you to process audio files and obtain complete music-related features such as BPM, mood, key, timbre, and more.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
- [Endpoints](#endpoints)
- [Contributing](#contributing)
- [License](#license)

## Features

- Process audio files to extract music-related features.
- Analyze audio files to determine BPM, mood, key, timbre, and more.
- Store analysis results in JSON format.
- Provides a JSON file explaining how to read the data.
- Provides RESTful API endpoints to access analysis results.

## Getting Started

Follow these instructions to get the project up and running on your local machine.

## Requirements

Before you begin, follow these steps:
- create a virtual environment
     ```bash
    python -m venv venv

- activate the virtual environment (on Windows)
     ```bash
    venv\Scripts\activate

- Install packages from requirements.txt
     ```bash
    pip install -r requirements.txt

### Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/albedimaria/backend_thesis.git
   
### Usage

- Start the Flask application:
     ```bash
     python main.py

- Access the application in your web browser at http://localhost:5000.
- Select the audio folder containing files for processing and retrieve the analysis results.

### Endpoints

The applicationprovides the following endpoint:
- `/process_audio`: process audio files and retrieve analysis results.

### Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

Fork the repository.
Create a new branch for your feature or bug fix.
Make your changes and commit them.
Push your changes onto your fork.
Open a pull request.

### License
This project is licensed under the MIT License. See the LICENSE file for details.

