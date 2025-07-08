<h1 align="center">
  ollamacord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

ollamacord transforms Discord into a collaborative LLM frontend.

## Features

### Reply-based chat system:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

---

### Model switching with `/model`:
![image](https://github.com/user-attachments/assets/9fbb9f56-9004-4997-a864-5b2ec67bac8f)

run local models with:
- [Ollama](https://ollama.com)

---

### And more:
- Supports image attachments when using a vision model (like gpt-4.1, claude-4, llama-4, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- 1 Python file, ~200 lines of code

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/datapush3r/ollamacord
