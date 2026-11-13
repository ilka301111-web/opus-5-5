import asyncio
import html
import logging
import os
import re

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import BufferedInputFile, Message
from aiogram.utils.chat_action import ChatActionSender
from aiohttp import web
from openai import AsyncOpenAI

# ======================= КОНФИГУРАЦИЯ =======================
# Значения берутся из переменных окружения (на Render ты задашь их в настройках).
# Для запуска на своём компьютере можно вместо os.getenv(...) вписать значения прямо сюда.
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("BASE_URL", "https://your-openai-compatible-endpoint/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "claude-opus-5-5")
ALLOWED_USER_ID = int(os.getenv("ALLOWED_USER_ID", "0"))  # твой Telegram ID (узнать у @userinfobot)

SYSTEM_PROMPT = "Ты полезный ассистент. Отвечай по существу."
MAX_HISTORY_MESSAGES = 40   # сколько последних сообщений хранить в контексте
CHUNK_LIMIT = 3500          # безопасный размер одной части (лимит Telegram — 4096)
MAX_PARTS = 4               # если частей больше — отправляем ответ файлом .md
# ============================================================

logging.basicConfig(level=logging.INFO)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
router = Router()
router.message.filter(F.from_user.id == ALLOWED_USER_ID)  # чужие сообщения игнорируются

history: list[dict] = []
history_lock = asyncio.Lock()


# ---------- Форматирование: Markdown -> безопасный Telegram HTML ----------
FENCE_RE = re.compile(r"```([\w+#.-]*)[ \t]*\n(.*?)```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
HEADING_RE = re.compile(r"^#{1,6}[ \t]+(.+)$", re.MULTILINE)


def _format_plain_part(text: str) -> str:
    """Форматирует текст вне блоков кода: inline-код, жирный, заголовки."""
    result = []
    pos = 0
    for m in INLINE_CODE_RE.finditer(text):
        result.append(_format_text_only(text[pos:m.start()]))
        result.append(f"<code>{html.escape(m.group(1))}</code>")
        pos = m.end()
    result.append(_format_text_only(text[pos:]))
    return "".join(result)


def _format_text_only(text: str) -> str:
    text = html.escape(text, quote=False)
    text = HEADING_RE.sub(r"<b>\1</b>", text)
    text = BOLD_RE.sub(r"<b>\1</b>", text)
    return text


def md_to_html(text: str) -> str:
    result = []
    pos = 0
    for m in FENCE_RE.finditer(text):
        result.append(_format_plain_part(text[pos:m.start()]))
        lang = m.group(1)
        code = html.escape(m.group(2).rstrip("\n"), quote=False)
        if lang:
            result.append(f'<pre><code class="language-{html.escape(lang)}">{code}</code></pre>')
        else:
            result.append(f"<pre>{code}</pre>")
        pos = m.end()
    result.append(_format_plain_part(text[pos:]))
    return "".join(result)


# ---------- Разбиение длинного текста (не ломая блоки кода) ----------
def split_markdown(text: str, limit: int = CHUNK_LIMIT) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    in_fence = False
    fence_lang = ""

    def flush():
        nonlocal current, current_len
        if not current:
            return
        body = "\n".join(current)
        if in_fence:
            body += "\n```"
        chunks.append(body)
        current = []
        current_len = 0
        if in_fence:
            current.append(f"```{fence_lang}")
            current_len = len(current[0]) + 1

    for line in text.split("\n"):
        # слишком длинные строки режем жёстко
        pieces = [line[i:i + limit - 20] for i in range(0, len(line), limit - 20)] or [""]
        for piece in pieces:
            if current_len + len(piece) + 1 > limit - 10:
                flush()
            current.append(piece)
            current_len += len(piece) + 1
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
            else:
                in_fence = True
                fence_lang = stripped[3:].strip()

    if current:
        chunks.append("\n".join(current))

    # убираем пустые части и пустые блоки кода, оставшиеся после переноса
    return [c for c in chunks if c.strip() and c.strip() not in ("```", f"```{fence_lang}")]


# ---------- Отправка ответа ----------
async def send_safe(message: Message, chunk: str) -> None:
    rendered = md_to_html(chunk)
    if len(rendered) <= 4096:
        try:
            await message.answer(rendered, parse_mode=ParseMode.HTML)
            return
        except TelegramBadRequest as e:
            logging.warning("HTML отклонён Telegram (%s), отправляю как обычный текст", e)
    await message.answer(chunk[:4096], parse_mode=None)


async def send_long(message: Message, text: str) -> None:
    text = text.strip() or "(пустой ответ)"
    chunks = split_markdown(text)

    if len(chunks) > MAX_PARTS:
        file = BufferedInputFile(text.encode("utf-8"), filename="answer.md")
        await message.answer_document(file, caption="Ответ слишком длинный — отправляю файлом.")
        return

    for chunk in chunks:
        await send_safe(message, chunk)


# ---------- Хендлеры ----------
@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Просто напиши сообщение — я передам его модели.\n"
        "/clear — очистить историю диалога.",
        parse_mode=None,
    )


@router.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    async with history_lock:
        history.clear()
    await message.answer("История диалога очищена.", parse_mode=None)


@router.message(F.text)
async def handle_text(message: Message, bot: Bot) -> None:
    async with history_lock:
        history.append({"role": "user", "content": message.text})
        del history[:-MAX_HISTORY_MESSAGES]
        while history and history[0]["role"] != "user":
            history.pop(0)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history]

        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logging.exception("Ошибка при обращении к API")
            history.pop()  # убираем сообщение, на которое не получили ответ
            await message.answer(f"Ошибка API: {e}"[:4000], parse_mode=None)
            return

        history.append({"role": "assistant", "content": answer})

    await send_long(message, answer)


@router.message()
async def handle_other(message: Message) -> None:
    await message.answer("Я понимаю только текстовые сообщения.", parse_mode=None)


async def health(request: web.Request) -> web.Response:
    return web.Response(text="ok")


async def start_web_server() -> None:
    """Мини-сервер для Render и UptimeRobot: отвечает 'ok' на запросы."""
    app = web.Application()
    app.router.add_get("/", health)
    app.router.add_get("/health", health)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", int(os.getenv("PORT", "10000")))
    await site.start()


async def main() -> None:
    bot = Bot(
        token=TELEGRAM_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher()
    dp.include_router(router)
    await start_web_server()
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
