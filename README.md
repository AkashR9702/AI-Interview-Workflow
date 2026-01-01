# AI Interview Workflow

Built an automated interview system using LangGraph that handles the whole interview process - from reading resumes to asking questions and giving feedback.

## What it does

- Reads PDF resumes and pulls out candidate info
- Takes a job description and figures out what to ask
- Generates relevant technical questions using LLMs
- Collects answers from candidates (with human-in-the-loop)
- Gives structured feedback with scores
- Uses LangGraph to manage the whole workflow

## Tech Stack

- Python 3.11
- LangChain for LLM stuff
- LangGraph for workflow management
- Pydantic for data validation
- Ollama for running LLMs locally
- PyPDF for reading resumes

## How the code is organized
```
Pre_Int_Agent/
├── Interview_agent.py          # Everything in one file
├── requirements.txt             # What you need to install
├── sample_cv.pdf                # Test resume
├── sample_jd.txt                # Test job description
└── modular_version/             # Cleaner version split into modules
    ├── config.py                # Settings
    ├── schemas.py               # Data structures
    ├── nodes.py                 # Main functions
    ├── graph.py                 # Workflow setup
    └── run.py                   # Run this
```

## Setting it up

1. Clone this repo
2. Install what you need:
```bash
pip install -r requirements.txt
```
3. Set up your LLM:
   - Using Ollama locally? Make sure it's running with llama3.2
   - Want to use OpenAI or Anthropic? Add your API key and change the config

## Running it

Simple version:
```bash
python Interview_agent.py
```

Modular version:
```bash
cd modular_version
python run.py
```

## How it works

1. Reads the resume PDF
2. Looks at the job description
3. Comes up with 5 good questions based on both
4. Asks for answers (that's the human-in-the-loop part)
5. Evaluates everything and gives scores with feedback

## The workflow keeps track of

- What's in the resume
- What the job needs
- Questions it generated
- Answers collected
- Final evaluation

## What's next

- Web interface for easier interaction
- Support for different LLM providers
- Audio/video interview capability
- Better evaluation metrics
- Resume storage and candidate tracking
