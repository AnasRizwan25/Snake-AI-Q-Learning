# Snake AI Project

This project is a simple Snake game implemented in Python, with an AI agent trained to play using reinforcement learning.

## Overview

The game includes:
- a classic Snake gameplay loop
- a Q-learning based AI model
- saved training data in a `.npy` file
- a virtual environment for project dependencies

## Files

- `demo.py` – main game and AI logic
- `snake_q_table.npy` – trained Q-table used by the model
- `.gitignore` – ignores environment and generated files

## Goal

The aim of the project is to train an agent to learn how to play Snake and improve over time by maximizing its score while avoiding collisions.

## Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

Install the required libraries:

```bash
pip install numpy pygame
```

## Run

```bash
python demo.py
```

## Notes

This project is intended as a learning/demo project for Python game development and reinforcement learning basics.