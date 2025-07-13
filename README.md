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

* **Seamless Chat Experience**: Start a conversation by mentioning the bot and continue it with replies. The bot automatically quotes your message for context, making conversations easy to follow.
* **Dynamic Model Management**: The bot automatically detects your locally downloaded Ollama models.
* **Persistent Configuration**: Set a default model that persists even after restarting the bot.
* **Full Admin Control**: A suite of admin commands allows you to manage your Ollama instance directly from Discord, including listing, downloading, removing, and setting a default model.
* **Advanced Features**:
    * Supports image attachments with vision models.
    * Customizable system prompt to define the bot's personality.
    * Handles long responses by automatically splitting them into multiple messages.
    * Robust error handling for a smoother user experience.
    * Efficient and asynchronous, built on a single Python file.

## Instructions

1.  **Clone the repo:**
    ```bash
    git clone https://github.com/datapush3r/ollamacord
    cd ollamacord
    ```
2.  **Set up your configuration:**
    Create a copy of `config-example.yaml` and name it `config.yaml`. Fill in your Discord bot token and client ID. You can also optionally set a `default_model` to be used when the bot starts.
3.  **Run the bot:**
    * **With Docker (Recommended):**
        ```bash
        docker compose up -d
        ```
        This will build the container and run the bot in the background. The `docker-compose.yaml` is configured to mount the current directory, so your `config.yaml` is accessible to the bot.
    * **Without Docker:**
        ```bash
        python -m pip install -U -r requirements.txt
        python ollamacord.py
        ```

## Commands

* `/ollamacord help`: Shows a complete list of commands and features.
* `/ollamacord search <query>`: Searches the Ollama model library for new models to download.
* `/ollamacord download <model_name>`: Downloads a new model from the library to your local Ollama instance.
* `/ollamacord switch <model_name>`: Switches the active model to one of your locally downloaded models for the current session.

### Admin Commands

* `/ollamacord admin list`: Lists all models available in your local Ollama instance.
* `/ollamacord admin show <model_name>`: Shows detailed information about a specific local model.
* `/ollamacord admin ps`: Lists all models currently running in memory.
* `/ollamacord admin rm <model_name>`: Deletes a model from your local storage.
* `/ollamacord admin pull <model_name>`: An alias for the `/ollamacord download` command.
* `/ollamacord admin setdefault <model_name>`: Sets the default model in `config.yaml`, which will be used when the bot starts.

## Star History

<a href="https://star-history.com/#datapush3r/ollamacord&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=datapush3r/ollamacord&type=Date" />
  </picture>
</a>

