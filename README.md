<h1 align="center">
  Ollamacord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="Ollamacord in action">
</p>

Ollamacord transforms Discord into a collaborative frontend for local language models, powered by [Ollama](https://ollama.com).

This project is a fork of [llmcord](https://github.com/jakobdylanc/llmcord), redesigned to focus exclusively on Ollama for a more streamlined and powerful experience.

## Features

* **Seamless Chat Experience**: Start a conversation by mentioning the bot and continue it with replies.
* **Context-Aware Model Switching**: Automatically uses a vision or code model based on message content (images, code blocks).
* **Long-Term Memory**: Automatically builds a profile of each user's preferences and facts over time to provide more personalized responses.
* **Simplified Permissions**: All bot commands are managed by a single Discord role for easy setup.
* **Dynamic Model Management**: The bot automatically detects your locally downloaded Ollama models.
* **Persistent Configuration**: Set default, vision, and code models in a `config.yaml` file.
* **Full Command Suite**: A comprehensive set of commands allows you to manage your Ollama instance directly from Discord.

## Instructions

1.  **Clone the repo:**
    ```bash
    git clone [https://github.com/datapush3r/ollamacord](https://github.com/datapush3r/ollamacord)
    cd ollamacord
    ```
2.  **Set up your configuration:**
    Create a copy of `config-example.yaml` and name it `config.yaml`. Fill in your Discord bot token, client ID, and the `authorized_role_id`.
3.  **Run the bot:**
    * **With Docker (Recommended):**
        ```bash
        docker compose up -d --build
        ```
        This will build the container and run the bot in the background.
    * **Without Docker:**
        ```bash
        python -m pip install -U -r requirements.txt
        python ollamacord.py
        ```

## Commands

All commands are restricted to the `authorized_role_id` set in your `config.yaml`.

* `/ollamacord help`: Shows a complete list of commands and features.
* `/ollamacord search <query>`: Searches the Ollama model library.
* `/ollamacord download <model_name>`: Downloads a model to your local Ollama instance.
* `/ollamacord switch <model_name>`: Temporarily switches the active model for the current session.
* `/ollamacord reset`: Resets the session model and prompt to the defaults from `config.yaml`.
* `/ollamacord list`: Lists all locally available models.
* `/ollamacord show <model_name>`: Shows detailed information about a local model.
* `/ollamacord ps`: Lists all models currently running in memory.
* `/ollamacord rm <model_name>`: Deletes a model from your local storage.
* `/ollamacord setdefault <model_name>`: Sets the default model in `config.yaml`.
* `/ollamacord setvision <model_name>`: Sets the vision model in `config.yaml`.
* `/ollamacord setcode <model_name>`: Sets the code model in `config.yaml`.
* `/ollamacord setprompt <prompt>`: Adds to the system prompt for the current session.
* `/ollamacord getprompt`: Shows the current system prompt (base + session).
* `/ollamacord getmodels`: Shows the currently configured default, vision, and code models.

## Star History

<a href="https://star-history.com/#datapush3r/ollamacord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date" />
  </picture>
</a>

