import asyncio
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
import logging
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
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
MAX_MESSAGE_NODES = 500
OLLAMA_API_BASE_URL = "https://ollamadb.dev/api/v1"

# --- Config and State ---
def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        return yaml.safe_load(file)

config = get_config()
curr_model = None # Will be set in on_ready
msg_nodes = {}
last_task_time = 0

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

# --- Slash Commands ---
ollamacord_group = Group(name="ollamacord", description="Commands for the Ollamacord bot.")

@ollamacord_group.command(name="help", description="Shows the help message for the bot.")
async def help_command(interaction: discord.Interaction):
    embed = discord.Embed(title="Ollamacord Help", description="Here's how to use the bot:", color=discord.Color.blue())
    embed.add_field(name="How to Chat", value="To start a conversation, just `@mention` me in any channel. To continue the conversation, simply reply to my messages.", inline=False)
    embed.add_field(name="/ollamacord search", value="Search for new models to download from the Ollama library.", inline=False)
    embed.add_field(name="/ollamacord download", value="Download a model from the Ollama library to use with the bot.", inline=False)
    embed.add_field(name="/ollamacord switch", value="Switch between your locally downloaded models.", inline=False)
    embed.add_field(name="Admin Commands", value="Admins can use `/ollamacord admin` to `list`, `show`, `rm`, and `pull` local models.", inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@ollamacord_group.command(name="search", description="Search for models in the Ollama library.")
@discord.app_commands.autocomplete(query=search_models_autocomplete)
async def model_search(interaction: discord.Interaction, query: str):
    await interaction.response.defer(ephemeral=True)
    try:
        response = await httpx_client.get(f"{OLLAMA_API_BASE_URL}/models", params={"search": query, "limit": 10})
        response.raise_for_status()
        data = response.json()
        if not data.get("models"):
            await interaction.followup.send("No models found for your query.", ephemeral=True)
            return
        embed = discord.Embed(title=f"Search Results for '{query}'", color=discord.Color.blue())
        for model in data["models"]:
            name = f"{model.get('model_identifier', 'N/A')}"
            description = model.get('description', 'No description available.')
            pulls = model.get('pulls', 0)
            embed.add_field(name=name, value=f"**Pulls:** {pulls:,}\n{description}", inline=False)
        await interaction.followup.send(embed=embed, ephemeral=True)
    except httpx.HTTPError as e:
        await interaction.followup.send(f"An error occurred while searching for models: {e}", ephemeral=True)

@ollamacord_group.command(name="download", description="Download a model from the Ollama library.")
@discord.app_commands.autocomplete(model_name=search_models_autocomplete)
async def model_download(interaction: discord.Interaction, model_name: str):
    await interaction.response.defer(ephemeral=False)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        await interaction.followup.send(f"Starting download for model `{model_name}`. This may take a while...")
        async with httpx.AsyncClient(base_url=ollama_base_url, timeout=None) as ollama_client:
            pull_response = await ollama_client.post("/api/pull", json={"name": model_name})
            pull_response.raise_for_status()
        await interaction.followup.send(f"Model `{model_name}` has been downloaded successfully.")
    except httpx.HTTPStatusError as e:
        await interaction.followup.send(f"An HTTP error occurred: {e}", ephemeral=False)
    except Exception as e:
        await interaction.followup.send(f"An unexpected error occurred: {e}", ephemeral=False)

@ollamacord_group.command(name="switch", description="Switch the current active model.")
@discord.app_commands.autocomplete(model=local_ollama_model_autocomplete)
async def model_switch(interaction: discord.Interaction, model: str):
    global curr_model
    if model == curr_model:
        output = f"Current model is already `{model}`."
    else:
        curr_model = model
        output = f"Model switched to: `{model}`"
        logging.info(output)
    await interaction.response.send_message(output, ephemeral=True)

admin_group = Group(name="admin", description="Admin commands for managing the local Ollama instance.", parent=ollamacord_group)

async def is_admin(interaction: discord.Interaction) -> bool:
    current_config = await asyncio.to_thread(get_config)
    if interaction.user.id not in current_config["permissions"]["users"]["admin_ids"]:
        await interaction.response.send_message("You don't have permission to use this admin command.", ephemeral=True)
        return False
    return True

@admin_group.command(name="list", description="List all models available locally.")
async def admin_list(interaction: discord.Interaction):
    if not await is_admin(interaction): return
    await interaction.response.defer(ephemeral=True)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.get("/api/tags")
            response.raise_for_status()
        models = response.json().get("models", [])
        if not models:
            await interaction.followup.send("No local models found.", ephemeral=True)
            return
        embed = discord.Embed(title="Local Ollama Models", color=discord.Color.blue())
        description = "\n".join([f"**{model.get('name')}** ({model.get('size', 0) / (1024**3):.2f} GB)" for model in models])
        embed.description = description
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)

@admin_group.command(name="show", description="Show detailed information for a local model.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def admin_show(interaction: discord.Interaction, model_name: str):
    if not await is_admin(interaction): return
    await interaction.response.defer(ephemeral=True)
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
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)

@admin_group.command(name="ps", description="List running models on Ollama.")
async def admin_ps(interaction: discord.Interaction):
    if not await is_admin(interaction): return
    await interaction.response.defer(ephemeral=True)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.get("/api/ps")
            response.raise_for_status()
        models = response.json().get("models", [])
        if not models:
            await interaction.followup.send("No models are currently running.", ephemeral=True)
            return
        embed = discord.Embed(title="Running Ollama Models", color=discord.Color.green())
        description = "\n".join([f"**{m.get('name')}** ({m.get('size', 0) / (1024**3):.2f} GB)" for m in models])
        embed.description = description
        await interaction.followup.send(embed=embed, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}\n(Note: This command may require a recent version of Ollama.)", ephemeral=True)

@admin_group.command(name="rm", description="Remove a local model.")
@discord.app_commands.autocomplete(model_name=local_ollama_model_autocomplete)
async def admin_rm(interaction: discord.Interaction, model_name: str):
    if not await is_admin(interaction): return
    await interaction.response.defer(ephemeral=True)
    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.delete("/api/delete", json={"name": model_name})
            response.raise_for_status()
        await interaction.followup.send(f"Successfully removed model `{model_name}`.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"An error occurred: {e}", ephemeral=True)

@admin_group.command(name="pull", description="Pull a model from a registry.")
@discord.app_commands.autocomplete(model_name=search_models_autocomplete)
async def admin_pull(interaction: discord.Interaction, model_name: str):
    if not await is_admin(interaction): return
    await model_download.callback(interaction, model_name)

# --- Bot Events ---
@discord_bot.event
async def on_ready() -> None:
    global curr_model
    discord_bot.tree.add_command(ollamacord_group)
    if client_id := config.get("client_id"):
        logging.info(f"\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id={client_id}&permissions=412317273088&scope=bot\n")
    
    # Set the initial model
    try:
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
        logging.error(f"Could not connect to Ollama to set initial model: {e}")

    await discord_bot.tree.sync()

@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time
    is_dm = new_msg.channel.type == discord.ChannelType.private

    if (not is_dm and discord_bot.user not in new_msg.mentions) or new_msg.author.bot:
        return

    if not curr_model:
        await new_msg.channel.send("I can't respond right now because no Ollama models are available. An admin can download one using `/ollamacord download`.")
        return

    config = await asyncio.to_thread(get_config)
    permissions = config["permissions"]
    user_is_admin = new_msg.author.id in permissions["users"]["admin_ids"]
    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(filter(None, (new_msg.channel.id, getattr(new_msg.channel, "parent_id", None), getattr(new_msg.channel, "category_id", None))))

    (allowed_user_ids, blocked_user_ids), (allowed_role_ids, blocked_role_ids), (allowed_channel_ids, blocked_channel_ids) = (
        (perm["allowed_ids"], perm["blocked_ids"]) for perm in (permissions["users"], permissions["roles"], permissions["channels"])
    )
    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = user_is_admin or allow_all_users or new_msg.author.id in allowed_user_ids or any(id in allowed_role_ids for id in role_ids)
    is_bad_user = not is_good_user or new_msg.author.id in blocked_user_ids or any(id in blocked_role_ids for id in role_ids)

    allow_dms = config.get("allow_dms", True)
    allow_all_channels = not allowed_channel_ids
    is_good_channel = user_is_admin or allow_dms if is_dm else allow_all_channels or any(id in allowed_channel_ids for id in channel_ids)
    is_bad_channel = not is_good_channel or any(id in blocked_channel_ids for id in channel_ids)

    if is_bad_user or is_bad_channel:
        return

    model = curr_model
    model_parameters = None # Simplified

    try:
        ollama_base_url = config["ollama_base_url"].removesuffix("/v1")
        async with httpx.AsyncClient(base_url=ollama_base_url) as ollama_client:
            response = await ollama_client.post("/api/show", json={"name": model}, timeout=5)
            if response.status_code == 404:
                await new_msg.channel.send(f"Model `{model}` not found locally. You can try to download it with `/ollamacord download`.")
                return
            response.raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        logging.exception(f"Failed to communicate with Ollama API: {e}")
        await new_msg.channel.send("Could not connect to the Ollama API. Please check if it's running.")
        return

    openai_client = AsyncOpenAI(base_url=config["ollama_base_url"], api_key="sk-no-key-required")

    accept_images = any(x in model.lower() for x in VISION_MODEL_TAGS)
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

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        system_prompt = system_prompt.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()
        messages.append(dict(role="system", content=system_prompt))

    curr_content = finish_reason = edit_task = None
    response_msgs = []
    response_contents = []
    
    quoted_content = new_msg.content
    if quoted_content.lower().startswith("user:"):
        quoted_content = quoted_content[len("user:"):].lstrip()
    reply_quote = f"**Replying to {new_msg.author.display_name}**\n> {quoted_content[:250]}\n\n"
    
    embed = discord.Embed(color=EMBED_COLOR_INCOMPLETE)
    for warning in sorted(user_warnings):
        embed.add_field(name=warning, value="", inline=False)

    use_plain_responses = config.get("use_plain_responses", False)
    max_message_length = 2000 if use_plain_responses else (4096 - len(STREAMING_INDICATOR) - len(reply_quote))

    try:
        async with new_msg.channel.typing():
            async for curr_chunk in await openai_client.chat.completions.create(model=model, messages=messages[::-1], stream=True, extra_body=model_parameters):
                if finish_reason is not None: break
                if not (choice := curr_chunk.choices[0] if curr_chunk.choices else None): continue
                finish_reason = choice.finish_reason
                prev_content = curr_content or ""
                curr_content = choice.delta.content or ""
                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "": continue

                if start_next_msg := response_contents == [] or len(response_contents[-1] + new_content) > max_message_length:
                    response_contents.append("")
                response_contents[-1] += new_content

                if not use_plain_responses:
                    ready_to_edit = (edit_task is None or edit_task.done()) and datetime.now().timestamp() - last_task_time >= EDIT_DELAY_SECONDS
                    msg_split_incoming = finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                    is_final_edit = finish_reason is not None or msg_split_incoming
                    is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                    if start_next_msg or ready_to_edit or is_final_edit:
                        if edit_task is not None: await edit_task
                        
                        description = response_contents[-1] if is_final_edit else (response_contents[-1] + STREAMING_INDICATOR)
                        cleaned_description = re.sub(r"<think>.*?</think>", "", description, flags=re.DOTALL).strip()
                        
                        embed.description = reply_quote + cleaned_description
                        embed.color = EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                        embed.set_footer(text=f"Model: {model}")

                        if start_next_msg:
                            reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                            response_msg = await reply_to_msg.reply(embed=embed, silent=True)
                            response_msgs.append(response_msg)
                            msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                            await msg_nodes[response_msg.id].lock.acquire()
                        else:
                            edit_task = asyncio.create_task(response_msgs[-1].edit(embed=embed))
                        last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                cleaned_contents = [re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip() for content in response_contents]
                for content in cleaned_contents:
                    if not content: continue
                    reply_to_msg = new_msg if not response_msgs else response_msgs[-1]
                    response_msg = await reply_to_msg.reply(content=f"{content}\n\n*Model: {model}*", suppress_embeds=True)
                    response_msgs.append(response_msg)
                    msg_nodes[response_msg.id] = MsgNode(parent_msg=new_msg)
                    await msg_nodes[response_msg.id].lock.acquire()
    except Exception as e:
        logging.exception("Error while generating response")
        await new_msg.channel.send(f"Sorry, an error occurred. Please try again.\nIf the problem persists, you might find help with `/ollamacord help`.")

    for response_msg in response_msgs:
        if response_msg.id in msg_nodes and msg_nodes[response_msg.id].lock.locked():
            full_text = "".join(response_contents)
            cleaned_text = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip()
            msg_nodes[response_msg.id].text = cleaned_text
            msg_nodes[response_msg.id].lock.release()

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            if msg_id in msg_nodes:
                async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                    msg_nodes.pop(msg_id, None)

async def main() -> None:
    await discord_bot.start(config["bot_token"])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
