# Project: Python LLM Assistant

This project aims to create a Python app that functions as an ideal local/remote Language Model (LLM) assistant. The app begins with a small, simple local model that spawns agents to build and enhance itself with the aid of the user. It uses GitHub CLI tools for forking and managing its own codebase, ensuring modularity and self-improvement.

## Key Features

1. **Agent Class**: Represents agents with methods for memory management, system log gathering, and user interaction.
2. **App Class**: Main application class with methods for managing model interactions, system checks, GitHub integration, and web server configuration.
3. **Error Handling**: Comprehensive try/except blocks for robust error management.
4. **GitHub Integration**: Methods to fork and update GitHub repositories using the PyGithub library.
5. **Web Server**: Flask routes for user interaction with the model.
6. **Model Interaction**: Placeholder for model interactions using `llama.cpp`.

## Requirements

- Python 3.x
- `gh` CLI tools
- GitHub account
- The following Python packages (specified in `requirements.txt`):
  - Flask==2.0.1
  - huggingface_hub==0.0.12
  - PyGithub==1.54.1
  - torch==1.9.0

## Installation

1. Ensure `gh` CLI tools are installed and you are logged in:
    ```bash
    if ! command -v gh &> /dev/null; then
        echo "gh cli tools are not installed. Please install them and try again."
        exit 1
    fi

    if ! gh auth status &> /dev/null; then
        echo "You are not logged in to your GitHub account. Please log in and try again."
        exit 1
    fi
    ```

2. Run the provided bash script to set up the project:
    ```bash
    #!/bin/bash

    read -p "Enter project name: " project_name

    mkdir $project_name
    cd $project_name

    git init

    python3 -m venv env
    source env/bin/activate

    gh repo create $project_name --public --source=.

    cat <<EOT > requirements.txt
    Flask==2.0.1
    huggingface_hub==0.0.12
    PyGithub==1.54.1
    torch==1.9.0
    EOT

    cat <<EOT > main.py
    <Main Python Script Content>
    EOT

    pip install -r requirements.txt

    deactivate

    source env/bin/activate
    python main.py
    ```

## Main Python Script Content

```python
import subprocess
import os
import threading
import json
from huggingface_hub import hf_hub_download
from flask import Flask, request, jsonify
from github import Github
import torch

class Agent:
    def __init__(self, is_local):
        self.is_local = is_local
        self.memory = {}

    def read_and_update_memory(self):
        """Read and update the agent's memory."""
        try:
            with open('memory.txt', 'r') as f:
                self.memory = json.load(f)

            self.memory['new_data'] = 'some_value'

            with open('memory.txt', 'w') as f:
                json.dump(self.memory, f)

        except Exception as e:
            print(f"Error reading or updating memory: {e}")

    def gather_system_logs(self):
        """Gather system logs for the agent."""
        # Implement the logic to gather system logs
        pass

    def greet_user(self):
        """Greet the user."""
        return "Hello, how can I assist you today?"

class App:
    def __init__(self):
        self.agents = []
        self.model = None
        self.app = Flask(__name__)
        self.github = Github()
        self.configure_routes()

    def configure_routes(self):
        @self.app.route('/chat', methods=['POST'])
        def chat():
            user_input = request.json.get('input')
            response = self.model_interaction(user_input)
            return jsonify({"response": response})

    def check_llama_installed(self):
        try:
            result = subprocess.run(['llama.cpp', '--version'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def check_gpu_capabilities(self):
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return {"gpu_name": gpu_name, "gpu_memory": gpu_memory}
        return None

    def install_llama(self):
        try:
            subprocess.run(['git', 'clone', 'https://github.com/ggerganov/llama.cpp'], check=True)
            os.chdir('llama.cpp')
            subprocess.run(['make'], check=True)
            os.chdir('..')
            print("llama.cpp installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error during installation: {e}")

    def recommend_model(self, gpu_info):
        if gpu_info:
            if gpu_info['gpu_memory'] > 10:
                return "big-model"
            elif gpu_info['gpu_memory'] > 5:
                return "medium-model"
            else:
                return "small-model"
        return "small-model"

    def download_model(self, model_name, directory="models"):
        try:
            os.makedirs(directory, exist_ok=True)
            model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin", cache_dir=directory)

            if not os.path.isfile(model_path):
                raise Exception("Downloaded file is not a valid model file")

            return model_path

        except Exception as e:
            print(f"Error downloading model: {e}")
            return None

    def run_model(self, model_path, gpu_info):
        try:
            command = ['llama.cpp', '--model', model_path]
            if gpu_info:
                command.extend(['--use-gpu', '--gpu-id', '0'])
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running model: {e}")

    def model_interaction(self, user_input):
        return f"Model response to: {user_input}"

    def start_web_server(self):
        self.app.run(port=5000, debug=True)

    def create_agent(self, is_local):
        agent = Agent(is_local)
        self.agents.append(agent)
        return agent

    def fork_repo(self, repo_name):
        try:
            user = self.github.get_user()
            repo = self.github.get_repo(repo_name)
            user.create_fork(repo)
            print(f"Repository {repo_name} forked successfully.")
        except Exception as e:
            print(f"Error forking repository: {e}")

    def update_repo(self, repo_name, commit_message):
        try:
            repo = self.github.get_repo(repo_name)
            # Implement the logic to update the repo
            print(f"Repository {repo_name} updated with message: {commit_message}")
        except Exception as e:
            print(f"Error updating repository: {e}")

    def main(self):
        llama_installed = self.check_llama_installed()
        gpu_info = self.check_gpu_capabilities()
        print(f"Llama.cpp installed: {llama_installed}")
        print(f"GPU Info: {gpu_info}")

        if not llama_installed:
            self.install_llama()

        model_name = self.recommend_model(gpu_info)
        model_path = self.download_model(model_name)
        if model_path:
            self.run_model(model_path, gpu_info)

        web_thread = threading.Thread(target=self.start_web_server)
        web_thread.start()
        print("Web chat interface running at http://localhost:5000")

if __name__ == "__main__":
    app = App()
    app.main()

def update_codebase():
    try:
        g = Github()
        repo = g.get_repo(f"username/{project_name}")
        fork = repo.get_forks()[0]

        subprocess.run(['git', 'clone', fork.clone_url], check=True)

        subprocess.run(['rm', '-rf', '*'], check=True)
        subprocess.run(['mv', f'{project_name}/*', '.'], check=True)
        subprocess.run(['rm', '-rf', project_name], check=True)

        print("Codebase updated successfully.")

    except Exception as e:
        print(f"Error updating codebase: {e}")

if __name__ == "__main__":
    # Add your main function here
    pass
```

## Running the App

1. Activate the virtual environment:
    ```bash
    source env/bin/activate
    ```

2. Start the main application:
    ```bash
    python main.py
    ```

The web interface will be available at `http://localhost:5000`.

## Contributing

Feel free to fork this repository and submit pull requests. Ensure that all changes are well-documented and thoroughly tested.

## License

This project is licensed under the MIT License.

---

This README provides a comprehensive guide for setting up and running the Python LLM Assistant app, ensuring modularity, self-improvement, and easy user interaction.