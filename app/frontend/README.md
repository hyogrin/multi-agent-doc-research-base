# Gradio Chatbot Interface

This project implements a simple Gradio user interface to interact with the `/chat` API of the Microsoft Plan and Search Chatbot. The interface allows users to send messages and receive responses from the chatbot in real-time.

## Project Structure

```
frontend
├── src
│   ├── app.py            # Main entry point for the web application
│   └── config.py         # Configuration settings for the application 
├── .env.example          # Template for environment variables
├── pyproject.toml        # List of dependencies required for the project
└── README.md             # Documentation for the project
└── run_app.sh            # Script to run the application
```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- [uv](https://github.com/astral-sh/uv) package manager
- Azure subscription with OpenAI service enabled
- uv
```bash
uv venv .venv --python 3.11 --seed
source .venv/bin/activate
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plan-search-chatbot.git
   cd plan-search-chatbot/app/backend
   ```

2. Set up environment variables:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file and add your backend URL:
   ```
   API_URL=http://localhost:8000/chat

   ```

3. Install backend dependencies using uv:
   ```bash
   uv pip install -e .
   ```
   
4. Run the application:
   ```bash
   ./run_app.sh
   ```

## Usage

- Open your web browser and navigate to public URL `https://xxxxxxxxx.gradio.live` to access the Gradio interface.
- Enter your message in the input box and click "Submit" to interact with the chatbot.

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements for the project.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.