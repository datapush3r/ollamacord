import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import re
from typing import Any, Literal, Optional

import discord
from discord.app_commands import Choice, Group
from discord.ext import commands
import httpx
from openai import AsyncOpenAI
import yaml

# --- Basic Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

VISION_MODEL_TAGS = ("gemma", "llama", "pixtral", "mistral", "vision", "vl")
EMBED_COLOR_COMPLETE = discord.Color.dark_green()
MAX_MESSAGE_NODES = 500
OLLAMA_API_BASE_URL = "https://ollamadb.dev/api/v1"
CODE_FILE_EXTENSIONS = (
    ".py", ".js", ".ts", ".html", ".css", ".scss", ".java", ".c", ".cpp", ".cs",
    ".go", ".rb", ".php", ".rs", ".swift", ".kt", ".lua", ".pl", ".sh", ".json",
    ".yaml", ".yml", ".md", ".sql", ".dockerfile", "Dockerfile", ".xml", ".bat"
)


# --- Config and State ---
def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    try:
        with open(filename, encoding="utf-8") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.critical(f"Config file '{filename}' not found. Please copy 'config-example.yaml' to '{filename}' and fill in your details.")
        exit()
    except yaml.YAMLError as e:
        logging.critical(f"Error parsing YAML from '{filename}': {e}")
        exit()

def save_config(data: dict[str, Any], filename: str = "config.yaml"):
    with open(filename, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)

config = get_config()
curr_model = None # Will be set in on_ready as the global default
session_prompt = None # Stores the system prompt for the current session
msg_nodes = {}

# --- Bot Initialization ---
intents = discord.Intents.default()
intents.message_content = True
activity = discord.CustomActivity(name=(config.get("status_message") or "github.com/datapush3r/ollamacord")[:128])
discord_bot = commands.Bot(intents=intents, activity=activity, command_prefix=None)
httpx_client = httpx.AsyncClient()

# --- Data Classes ---
@dataclass
class MsgNode:
    text: Optional[str] = None
    images: list[dict[str, Any]] = field(default_factory=list)
    role: Literal["user", "assistant"] = "assistant"
    user_id: Optional[int] = None
    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False
    parent_msg: Optional[discord.Message] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

# --- Autocomplete Functions ---
async def search_models_autocomplete(interaction: discord.Interaction, current: str) -> list[Choice[str]]:
    if not current: return []
    try:
        response = await httpx_client.get(f"{OLLAMA_API_BASE_URL}/models", params={"search": current, "limit": 25})
        response.raise_for_status()
        data = response.json()
        return [Choice(name=f"{m.get('model_name', 'N/A')} ({m.get('pulls', 0):,})", value=m.get('model_identifier', '')) for m in data.get("models", [])][:25]
    except httpx.HTTPError as e:
        logging.error(f"Error during model autocomplete search: {e}")
        return []

async def local_ollama_model_autocomplete(interaction: discord.Interaction, current: str) -> list[Choice[str]]:
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as client:
            response = await client.get("/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        return [Choice(name=m.get("name"), value=m.get("name")) for m in models if current.lower() in m.get("name", "").lower()][:25]
    except Exception as e:
        logging.error(f"Error fetching local ollama models: {e}")
        return []

# --- Permission Check ---
async def is_authorized(interaction: discord.Interaction) -> bool:
    current_config = await asyncio.to_thread(get_config)
    
    # If in a DM, check if DMs are allowed
    if interaction.guild is None:
        if current_config.get("allow_dms", True):
            return True
        else:
            await interaction.response.send_message("This command is not available in DMs.", ephemeral=True)
            return False

    # If in a server, check for the authorized role
    authorized_role_id = current_config.get("authorized_role_id")
    if not authorized_role_id:
        await interaction.response.send_message("The bot has not been configured with an authorized role. Please contact an administrator.", ephemeral=True)
        return False

    if not hasattr(interaction.user, 'roles'):
        return False

    user_roles = [role.id for role in interaction.user.roles]
    if authorized_role_id not in user_roles:
        await interaction.response.send_message("You do not have the required role to use this command.", ephemeral=True)
        return False
        
    return True

# --- Slash Commands ---
ollamacord_group = Group(name="ollamacord", description="Commands for the Ollamacord bot.")

@ollamacord_group.command(name="help", description="Shows the help message for the bot.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Ollamacord Help", description="Here's how to use the bot:", color=discord.Color.blue())
    embed.add_field(name="How to Chat", value="To start a conversation, just `@mention` me in any channel. To continue the conversation, simply reply to my messages.", inline=False)
    embed.add_field(name="General Commands", value="`/ollamacord search` - Search for new models.\n`/ollamacord download` - Download a model.\n`/ollamacord switch` - Temporarily switch the active model.", inline=False)
    embed.add_field(name="Management Commands", value="`/ollamacord list` - List local models.\n`/ollamacord show` - Show model details.\n`/ollamacord ps` - List running models.\n`/ollamacord rm` - Remove a local model.\n`/ollamacord setdefault` - Set the default model.\n`/ollamacord setvision` - Set the vision model.\n`/ollamacord setcode` - Set the code model.\n`/ollamacord setprompt` - Add to the system prompt for the current session.\n`/ollamacord getprompt` - Show the current system prompt.", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=False)

@ollamacord_group.command(name="search", description="Search for models in the Ollama library.")
@discord.app_commands.autocomplete(query=search_models_autocomplete)
async def model_search(interaction: discord.Interaction, query: str):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        response = await httpx_client.get(f"{OLLAMA_API_BASE_URL}/models", params={"search": query, "limit": 10})
        response.raise_for_status()
        data = response.json()
        if not data.get("models"):
            await interaction.followup.send("No models found for your query.")
            return
        embed = discord.Embed(title=f"Search Results for '{query}'", color=discord.Color.blue())
        for model in data["models"]:
            name = f"{model.get('model_identifier', 'N/A')}"
            description = model.get('description', 'No description available.')
            pulls = model.get('pulls', 0)
            embed.add_field(name=name, value=f"**Pulls:** {pulls:,}\n{description}", inline=False)
        await interaction.followup.send(embed=embed)
    except httpx.RequestError as e:
        await interaction.followup.send(f"An error occurred while connecting to the Ollama library: {e}")
    except httpx.HTTPStatusError as e:
        await interaction.followup.send(f"The Ollama library returned an error: {e.response.status_code}")


@ollamacord_group.command(name="download", description="Download a model from the Ollama library.")
@discord.app_commands.autocomplete(model_name=search_models_autocomplete)
async def model_download(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        await interaction.followup.send(f"Starting download for model `{model_name}`. This may take a while...")
        async with httpx.AsyncClient(base_url=ollama_base_url, timeout=None) as ollama_client:
            pull_response = await ollama_client.post("/api/pull", json={"name": model_name})
            pull_response.raise_for_status()
        await interaction.followup.send(f"Model `{model_name}` has been downloaded successfully.")
    except httpx.ConnectError:
        await interaction.followup.send(f"Could not connect to Ollama at {config['ollama_base_url']}. Please ensure it's running.")
    except httpx.HTTPStatusError as e:
        await interaction.followup.send(f"An HTTP error occurred while downloading: {e.response.text}")
    except Exception as e:
        await interaction.followup.send(f"An unexpected error occurred: {e}")

@ollamacord_group.command(name="switch", description="Temporarily switch the active model for this session.")
@discord.app_commands.autocomplete(model=local_ollama_model_autocomplete)
async def model_switch(interaction: discord.Interaction, model: str):
    if not await is_authorized(interaction): return
    global curr_model
    curr_model = model
    output = f"Model for this session switched to: `{model}`"
    logging.info(f"Session model switched to: {model} by {interaction.user.id}")
    await interaction.response.send_message(output, ephemeral=False)

async def set_model_in_config(interaction: discord.Interaction, model_name: str, model_type: str):
    await interaction.response.defer(ephemeral=False)
    try:
        current_config = await asyncio.to_thread(get_config)
        if 'model_settings' not in current_config:
            current_config['model_settings'] = {}
        current_config['model_settings'][model_type] = model_name
        await asyncio.to_thread(save_config, current_config)
        
        if model_type == 'default_model':
            global curr_model
            curr_model = model_name
            
        await interaction.followup.send(f"{model_type.replace('_', ' ').title()} has been set to `{model_name}`.")
    except IOError as e:
        await interaction.followup.send(f"An error occurred while writing to the config file: {e}")
    except Exception as e:
        await interaction.followup.send(f"An unexpected error occurred: {e}")

@ollamacord_group.command(name="setdefault", description="Set the default model in the config file.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def setdefault(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await set_model_in_config(interaction, model_name, "default_model")

@ollamacord_group.command(name="setvision", description="Set the vision model in the config file.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def setvision(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await set_model_in_config(interaction, model_name, "vision_model")

@ollamacord_group.command(name="setcode", description="Set the code model in the config file.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def setcode(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await set_model_in_config(interaction, model_name, "code_model")

@ollamacord_group.command(name="setprompt", description="Add to the system prompt for the current session.")
async def setprompt(interaction: discord.Interaction, prompt: str):
    if not await is_authorized(interaction): return
    global session_prompt
    session_prompt = prompt
    await interaction.response.send_message("The system prompt for this session has been updated.", ephemeral=False)

@ollamacord_group.command(name="getprompt", description="Show the current system prompt.")
async def getprompt(interaction: discord.Interaction):
    if not await is_authorized(interaction): return
    
    current_config = await asyncio.to_thread(get_config)
    base_prompt = current_config.get("system_prompt", "")
    session_specific_prompt = session_prompt or ""
    
    combined_prompt = f"{base_prompt}\n\n{session_specific_prompt}".strip()
    
    if not combined_prompt:
        await interaction.response.send_message("No system prompt is currently set.", ephemeral=False)
        return

    embed = discord.Embed(title="Current System Prompt", description=combined_prompt, color=discord.Color.blue())
    await interaction.response.send_message(embed=embed, ephemeral=False)

@ollamacord_group.command(name="list", description="List all models available locally.")
async def list_models(interaction: discord.Interaction):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.get("/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        if not models:
            await interaction.followup.send("No local models found.")
            return
        embed = discord.Embed(title="Local Ollama Models", color=discord.Color.blue())
        description = "\n".join([f"**{model.get('name')}** ({model.get('size', 0) / (1024**3):.2f} GB)" for model in models])
        embed.description = description
        await interaction.followup.send(embed=embed)
    except httpx.ConnectError:
        await interaction.followup.send(f"Could not connect to Ollama at {config['ollama_base_url']}.")
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")

@ollamacord_group.command(name="show", description="Show detailed information for a local model.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def show(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.post("/api/show", json={"name": model_name})
            response.raise_for_status()
        data = response.json()
        embed = discord.Embed(title=f"Details for {model_name}", color=discord.Color.blue())
        if modelfile := data.get("modelfile"):
            embed.add_field(name="Modelfile", value=f"```\n{modelfile[:1000]}\n```", inline=False)
        if details := data.get("details"):
            detail_str = "\n".join([f"**{k.replace('_', ' ').title()}**: {v}" for k,v in details.items()])
            embed.add_field(name="Details", value=detail_str, inline=False)
        await interaction.followup.send(embed=embed)
    except httpx.ConnectError:
        await interaction.followup.send(f"Could not connect to Ollama at {config['ollama_base_url']}.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            await interaction.followup.send(f"Model `{model_name}` not found locally.")
        else:
            await interaction.followup.send(f"An HTTP error occurred: {e.response.text}")
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")

@ollamacord_group.command(name="ps", description="List running models on Ollama.")
async def ps(interaction: discord.Interaction):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.get("/api/ps")
            response.raise_for_status()
        models = response.json().get("models", [])
        if not models:
            await interaction.followup.send("No models are currently running.")
            return
        embed = discord.Embed(title="Running Ollama Models", color=discord.Color.green())
        description = "\n".join([f"**{m.get('name')}** ({m.get('size', 0) / (1024**3):.2f} GB)" for m in models])
        embed.description = description
        await interaction.followup.send(embed=embed)
    except httpx.ConnectError:
        await interaction.followup.send(f"Could not connect to Ollama at {config['ollama_base_url']}.")
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}\n(Note: This command may require a recent version of Ollama.)")

@ollamacord_group.command(name="rm", description="Remove a local model.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def rm(interaction: discord.Interaction, model_name: str):
    if not await is_authorized(interaction): return
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.request("DELETE", "/api/delete", json={"name": model_name})
            response.raise_for_status()
        await interaction.followup.send(f"Successfully removed model `{model_name}`.")
    except httpx.ConnectError:
        await interaction.followup.send(f"Could not connect to Ollama at {config['ollama_base_url']}.")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            await interaction.followup.send(f"Model `{model_name}` not found, so it could not be removed.")
        else:
            await interaction.followup.send(f"An HTTP error occurred: {e.response.text}")
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}")

# --- Bot Events ---
@discord_bot.event
async def on_ready() -> None:
    global curr_model
    discord_bot.tree.add_command(ollamacord_group)
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")
    
    # Set the initial model
    try:
        model_settings = config.get("model_settings", {})
        default_model = model_settings.get("default_model")
        if default_model:
            curr_model = default_model
            logging.info(f"Default model set to: {curr_model}")
        else:
            ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
            async with httpx.AsyncClient(base_url=ollama_base_url) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
            models = response.json().get("models", [])
            if models:
                curr_model = models[0]['name']
                logging.info(f"Default model set to: {curr_model}")
            else:
                logging.warning("No local Ollama models found. Please download a model to use the bot.")
    except Exception as e:
        logging.error(f"Could not connect to Ollama to set initial model at {config.get('ollama_base_url')}: {e}")
        logging.warning("The bot will start, but will be unable to respond until a connection is established.")


    await discord_bot.tree.sync()

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return
        
    config = await asyncio.to_thread(get_config)
    
    # --- Permission Check ---
    if is_dm:
        if not config.get("allow_dms", True):
            return
    else: # It's a server message
        authorized_role_id = config.get("authorized_role_id")
        if not authorized_role_id:
            logging.warning("authorized_role_id is not set in config.yaml. The bot will not respond to messages.")
            return

        if not hasattr(new_msg.author, 'roles') or authorized_role_id not in [role.id for role in new_msg.author.roles]:
            return
    
    # --- Context-aware Model Selection ---
    model_settings = config.get("model_settings", {})
    default_model = model_settings.get("default_model", curr_model)
    vision_model = model_settings.get("vision_model", default_model)
    code_model = model_settings.get("code_model", default_model)
    
    model_to_use = default_model

    # Check for image attachments
    has_image = any(att.content_type and att.content_type.startswith("image") for att in new_msg.attachments)
    if has_image:
        if vision_model:
            model_to_use = vision_model
        else:
            await new_msg.channel.send("An image was attached, but no vision model is configured. Please ask an admin to set one with `/ollamacord setvision`.")
            return

    # Check for code
    has_code = "```" in new_msg.content or any(att.filename.endswith(ext) for ext in CODE_FILE_EXTENSIONS for att in new_msg.attachments)
    if not has_image and has_code:
        if code_model:
            model_to_use = code_model
        else:
            await new_msg.channel.send("Code was detected, but no code model is configured. Please ask an admin to set one with `/ollamacord setcode`.")
            return

    if not model_to_use:
        await new_msg.channel.send("I can't respond right now because no default model is configured. Please ask an admin to set one with `/ollamacord setdefault`.")
        return

    model_parameters = None # Simplified

    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.post("/api/show", json={"name": model_to_use}, timeout=5)
            if response.status_code == 404:
                await new_msg.channel.send(f"Model `{model_to_use}` not found locally. You can try to download it with `/ollamacord download`.")
                return
            response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logging.exception(f"Failed to communicate with Ollama API: {e}")
        await new_msg.channel.send("Could not connect to the Ollama API. Please check if it's running.")
        return

    openai_client = AsyncOpenAI(base_url=config["ollama_base_url"], api_key="sk-no-key-required")

    accept_images = any(x in model_to_use.lower() for x in VISION_MODEL_TAGS)
    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())
        async with curr_node.lock:
            if curr_node.text is None:
                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"
                text_parts = []
                if curr_node.role == "assistant":
                    if curr_msg.embeds:
                        full_description = curr_msg.embeds[0].description
                        parts = full_description.split('\n\n', 1)
                        if len(parts) > 1 and parts[0].startswith('**Replying to'):
                            text_parts.append(parts[1])
                        else:
                            text_parts.append(full_description)
                else:
                    cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()
                    if cleaned_content:
                        text_parts.append(cleaned_content)
                    text_parts.extend("\n".join(filter(None, (e.title, e.description, e.footer.text))) for e in curr_msg.embeds)

                good_attachments = [att for att in curr_msg.attachments if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))]
                attachment_responses = await asyncio.gather(*[httpx_client.get(att.url) for att in good_attachments])
                text_parts.extend(resp.text for att, resp in zip(good_attachments, attachment_responses) if att.content_type.startswith("text"))
                
                curr_node.text = "\n".join(text_parts)
                
                curr_node.images = [
                    dict(type="image_url", image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"))
                    for att, resp in zip(good_attachments, attachment_responses)
                    if att.content_type.startswith("image")
                ]
                curr_node.user_id = curr_msg.author.id if curr_node.role == "user" else None
                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)
                try:
                    if (
                        curr_msg.reference is None and discord_bot.user.mention not in curr_msg.content
                        and (prev_msg_in_channel := ([m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None])[0])
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author == (discord_bot.user if curr_msg.channel.type == discord.ChannelType.private else curr_msg.author)
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = is_public_thread and curr_msg.reference is None and curr_msg.channel.parent.type == discord.ChannelType.text
                        if parent_msg_id := curr_msg.channel.id if parent_is_thread_start else getattr(curr_msg.reference, "message_id", None):
                            if parent_is_thread_start:
                                curr_node.parent_msg = curr_msg.channel.starter_message or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                            else:
                                curr_node.parent_msg = curr_msg.reference.cached_message or await curr_msg.channel.fetch_message(parent_msg_id)
                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True
            
            if curr_node.images[:max_images]:
                content = ([dict(type="text", text=curr_node.text[:max_text])] if curr_node.text[:max_text] else []) + curr_node.images[:max_images]
            else:
                content = curr_node.text[:max_text]

            if content != "":
                message = dict(content=content, role=curr_node.role)
                messages.append(message)

            if len(curr_node.text) > max_text: user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images: user_warnings.add(f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message" if max_images > 0 else "⚠️ Can't see images")
            if curr_node.has_bad_attachments: user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    logging.info(f"Message received (user ID: {new_msg.author.id}, attachments: {len(new_msg.attachments)}, conversation length: {len(messages)}):\n{new_msg.content}")

    # Determine which system prompt to use
    base_prompt = config.get("system_prompt", "")
    session_specific_prompt = session_prompt or ""
    
    # Combine prompts, ensuring the base prompt is always included if it exists.
    combined_prompt = f"{base_prompt}\n\n{session_specific_prompt}".strip()

    if combined_prompt:
        now = datetime.now().astimezone()
        final_system_prompt = combined_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        messages.append(dict(role="system", content=final_system_prompt))
    
    quoted_content = new_msg.content
    if quoted_content.lower().startswith("user:"):
        quoted_content = quoted_content[len("user:"):].lstrip()
    reply_quote = f"**Replying to {new_msg.author.display_name}**\n> {quoted_content[:250]}\n\n"
    
    use_plain_responses = config.get("use_plain_responses", False)
    filter_thinking = config.get("filter_thinking_tags", True)
    max_message_length = 2000 if use_plain_responses else (4096 - len(reply_quote))

    try:
        async with new_msg.channel.typing():
            # Get the full response from the model
            completion = await openai_client.chat.completions.create(model=model_to_use, messages=messages[::-1], stream=False, extra_body=model_parameters)
            full_response_content = completion.choices[0].message.content or ""

            # Filter the full response if the setting is enabled
            if filter_thinking:
                final_content = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", full_response_content, flags=re.DOTALL).strip()
            else:
                final_content = full_response_content.strip()

            # Split the message into chunks if it's too long
            response_chunks = []
            if final_content:
                for i in range(0, len(final_content), max_message_length):
                    response_chunks.append(final_content[i:i + max_message_length])
            else: # Send a placeholder if the response is empty after filtering
                response_chunks.append("...")

            # Send the response chunks
            response_msgs = []
            for i, chunk in enumerate(response_chunks):
                reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                
                if use_plain_responses:
                    response_msg = await reply_to_msg.reply(content=chunk, silent=True)
                else:
                    embed = discord.Embed(color=EMBED_COLOR_COMPLETE)
                    if i == 0: # Only add the quote and warnings to the first message
                        embed.description = reply_quote + chunk
                        for warning in sorted(user_warnings):
                            embed.add_field(name=warning, value="", inline=False)
                    else:
                        embed.description = chunk
                    embed.set_footer(text=f"Model: {model_to_use}")
                    response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                
                response_msgs.append(response_msg)
                # Store node for conversation history
                msg_nodes[response_msg.id] = MsgNode(text=chunk, parent_msg=new_msg)

    except httpx.ConnectError:
        logging.exception("Connection to Ollama failed")
        await new_msg.channel.send("Sorry, I couldn't connect to Ollama. Please make sure it's running and accessible.")
    except httpx.HTTPStatusError as e:
        logging.exception(f"Ollama API returned an error: {e.response.status_code}")
        await new_msg.channel.send(f"Sorry, I received an error from the Ollama API (Status: {e.response.status_code}). Please check the model or the Ollama server.")
    except Exception as e:
        logging.exception("Error while generating response")
        await new_msg.channel.send(f"Sorry, an unexpected error occurred. Please try again.\nIf the problem persists, you might find help with `/ollamacord help`.")

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            if msg_id in msg_nodes:
                async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    msg_nodes.pop(msg_id, None)

async def main() -> None:
    await discord_bot.start(config["bot_token"])

if __name__ == "__main__":
    if not config.get("bot_token"):
        logging.critical("Bot token is missing from config.yaml. Please add it and try again.")
        exit()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    except httpx.ConnectError:
        logging.critical(f"Could not connect to Ollama at {config.get('ollama_base_url')}. Please ensure Ollama is running and accessible.")
    except Exception as e:
        logging.critical(f"An unexpected error occurred during startup: {e}")

