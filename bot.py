import asyncio
import base64
import html
import io
import logging
import os
import re
import zipfile
import xml.etree.ElementTree as ET

from aiogram import Bot, Dispatcher, F, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandStart
from aiogram.types import (BufferedInputFile, CallbackQuery, InlineKeyboardButton,
                           InlineKeyboardMarkup, Message)
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

MAX_FILE_BYTES = 20_000_000  # лимит Telegram на скачивание файлов ботом — 20 МБ
MAX_FILE_CHARS = int(os.getenv("MAX_FILE_CHARS", "300000"))  # сколько символов файла отправлять модели

# Мышление и длина ответа (настраиваются на Render в Environment):
# THINKING_BUDGET > 0 включает расширенное мышление с таким бюджетом токенов (0 = выключено).
# MAX_OUTPUT_TOKENS = 0 — не ограничивать длину ответа (решает провайдер).
THINKING_BUDGET = int(os.getenv("THINKING_BUDGET", "0"))
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "0"))
MAX_HISTORY_MESSAGES = 40   # сколько последних сообщений хранить в контексте
CHUNK_LIMIT = 3500          # безопасный размер одной части (лимит Telegram — 4096)
MAX_PARTS = 4               # если частей больше — отправляем ответ файлом .md
# ============================================================

logging.basicConfig(level=logging.INFO)

client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=900)
router = Router()
router.message.filter(F.from_user.id == ALLOWED_USER_ID)  # чужие сообщения игнорируются
router.callback_query.filter(F.from_user.id == ALLOWED_USER_ID)

history: list[dict] = []
history_lock = asyncio.Lock()

# Уровни мышления (бюджет токенов). Переключаются командой /lvl.
THINKING_LEVELS = {
    "off": ("Выкл", 0),
    "low": ("Низкое", 2000),
    "mid": ("Среднее", 8000),
    "high": ("Высокое", 16000),
    "max": ("Максимум", 32000),
}
state = {"thinking": THINKING_BUDGET, "brevity": "normal"}  # мышление: старт из THINKING_BUDGET

# Краткость ответа (команда /short). Мышление при этом не урезается:
# в запрос добавляется короткая инструкция только про длину итогового ответа.
BREVITY_LEVELS = {
    "normal": ("Обычно", ""),
    "short": (
        "Кратко",
        "Отвечай кратко и по делу: только суть, без вступлений, лишних пояснений и повторов. "
        "Если нужен код, приводи его полностью.",
    ),
    "tiny": (
        "Очень кратко",
        "Отвечай максимально кратко: одно-три предложения или только нужный код. "
        "Подробности давай, только если о них попросят.",
    ),
}


def brevity_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=("✅ " if state["brevity"] == key else "") + title,
            callback_data=f"short:{key}",
        )
        for key, (title, _) in BREVITY_LEVELS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def brevity_text() -> str:
    title = BREVITY_LEVELS[state["brevity"]][0]
    return (
        f"Длина ответа: {title}\n\n"
        "Мышление модели остаётся прежним (его уровень задаётся в /lvl), "
        "сокращается только итоговый ответ."
    )


def thinking_title() -> str:
    budget = state["thinking"]
    for title, value in THINKING_LEVELS.values():
        if value == budget:
            return f"{title} ({value} токенов)" if value else "Выкл"
    return f"свой бюджет ({budget} токенов)"


def thinking_keyboard() -> InlineKeyboardMarkup:
    buttons = [
        InlineKeyboardButton(
            text=("✅ " if state["thinking"] == value else "") + title,
            callback_data=f"think:{key}",
        )
        for key, (title, value) in THINKING_LEVELS.items()
    ]
    return InlineKeyboardMarkup(inline_keyboard=[buttons[:2], buttons[2:]])


def thinking_text() -> str:
    return (
        f"Мышление: {thinking_title()}\n\n"
        "Чем выше уровень, тем глубже думает модель, но тем больше уходит токенов и времени. "
        "Если после включения приходит «Ошибка API», выбери «Выкл»."
    )


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
        "Привет! Пришли текст, файл (код, txt, md, json, csv, PDF, DOCX, XLSX, PPTX) или фото.\n"
        "/clear — очистить историю диалога.\n"
        "/lvl — настроить глубину мышления модели.\n"
        "/short — сделать ответы короткими (мышление сохраняется).",
        parse_mode=None,
    )


@router.message(Command("short"))
async def cmd_short(message: Message) -> None:
    await message.answer(brevity_text(), reply_markup=brevity_keyboard(), parse_mode=None)


@router.callback_query(F.data.startswith("short:"))
async def on_brevity_choice(callback: CallbackQuery) -> None:
    key = callback.data.split(":", 1)[1]
    if key in BREVITY_LEVELS:
        state["brevity"] = key
    try:
        await callback.message.edit_text(brevity_text(), reply_markup=brevity_keyboard(), parse_mode=None)
    except TelegramBadRequest:
        pass  # выбран тот же режим — текст не изменился
    await callback.answer("Готово")


@router.message(Command("lvl"))
async def cmd_thinking(message: Message) -> None:
    await message.answer(thinking_text(), reply_markup=thinking_keyboard(), parse_mode=None)


@router.callback_query(F.data.startswith("think:"))
async def on_thinking_choice(callback: CallbackQuery) -> None:
    key = callback.data.split(":", 1)[1]
    if key in THINKING_LEVELS:
        state["thinking"] = THINKING_LEVELS[key][1]
    try:
        await callback.message.edit_text(thinking_text(), reply_markup=thinking_keyboard(), parse_mode=None)
    except TelegramBadRequest:
        pass  # выбран тот же уровень — текст не изменился
    await callback.answer("Готово")


@router.message(Command("clear"))
async def cmd_clear(message: Message) -> None:
    async with history_lock:
        history.clear()
    await message.answer("История диалога очищена.", parse_mode=None)


async def process_prompt(message: Message, bot: Bot, user_content, history_text: str | None = None) -> None:
    """user_content — строка или список частей (текст + изображение).
    history_text — что сохранить в историю вместо тяжёлого содержимого (например, фото)."""
    async with history_lock:
        history.append({"role": "user", "content": user_content})
        del history[:-MAX_HISTORY_MESSAGES]
        while history and history[0]["role"] != "user":
            history.pop(0)

        messages = list(history)
        instruction = BREVITY_LEVELS[state["brevity"]][1]
        if instruction:
            messages = [{"role": "system", "content": instruction}, *messages]

        kwargs: dict = {}
        budget = state["thinking"]
        if budget > 0:
            kwargs["extra_body"] = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            kwargs["max_tokens"] = budget + (MAX_OUTPUT_TOKENS or 16000)
        elif MAX_OUTPUT_TOKENS > 0:
            kwargs["max_tokens"] = MAX_OUTPUT_TOKENS

        try:
            async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
                response = await client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    **kwargs,
                )
            answer = response.choices[0].message.content or ""
        except Exception as e:
            logging.exception("Ошибка при обращении к API")
            history.pop()  # убираем сообщение, на которое не получили ответ
            hint = "\n\nПодсказка: мышление включено, попробуй /lvl → Выкл." if budget > 0 else ""
            await message.answer(f"Ошибка API: {e}{hint}"[:4000], parse_mode=None)
            return

        if history_text is not None:
            history[-1] = {"role": "user", "content": history_text}  # фото не храним — экономим токены
        history.append({"role": "assistant", "content": answer})

    await send_long(message, answer)


def decode_text(data: bytes) -> str | None:
    """Пытается прочитать файл как текст. Бинарные файлы возвращают None."""
    if b"\x00" in data[:2048]:
        return None
    for encoding in ("utf-8", "cp1251"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None


def extract_pdf(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
        return text.strip() or None
    except Exception:
        logging.exception("Не удалось прочитать PDF")
        return None


_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def extract_docx(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            root = ET.fromstring(z.read("word/document.xml"))
        paragraphs = ["".join(t.text or "" for t in p.iter(_W + "t")) for p in root.iter(_W + "p")]
        return "\n".join(paragraphs).strip() or None
    except Exception:
        logging.exception("Не удалось прочитать DOCX")
        return None


def extract_pptx(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            names = [n for n in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
            names.sort(key=lambda n: int(re.findall(r"\d+", n)[0]))
            parts = []
            for i, n in enumerate(names, 1):
                root = ET.fromstring(z.read(n))
                texts = [t.text or "" for t in root.iter(_A + "t")]
                parts.append(f"[Слайд {i}]\n" + "\n".join(texts))
        return "\n\n".join(parts).strip() or None
    except Exception:
        logging.exception("Не удалось прочитать PPTX")
        return None


def extract_xlsx(data: bytes) -> str | None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            shared: list[str] = []
            if "xl/sharedStrings.xml" in z.namelist():
                root = ET.fromstring(z.read("xl/sharedStrings.xml"))
                shared = ["".join(t.text or "" for t in si.iter(_S + "t")) for si in root.iter(_S + "si")]
            names = [n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n)]
            names.sort(key=lambda n: int(re.findall(r"\d+", n)[0]))
            parts = []
            for i, n in enumerate(names, 1):
                root = ET.fromstring(z.read(n))
                rows = []
                for row in root.iter(_S + "row"):
                    cells = []
                    for c in row.iter(_S + "c"):
                        v = c.find(_S + "v")
                        if c.get("t") == "s" and v is not None and v.text is not None:
                            cells.append(shared[int(v.text)] if int(v.text) < len(shared) else "")
                        elif c.get("t") == "inlineStr":
                            cells.append("".join(t.text or "" for t in c.iter(_S + "t")))
                        else:
                            cells.append(v.text if v is not None and v.text else "")
                    rows.append("\t".join(cells))
                parts.append(f"[Лист {i}]\n" + "\n".join(rows))
        return "\n\n".join(parts).strip() or None
    except Exception:
        logging.exception("Не удалось прочитать XLSX")
        return None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".gif")


async def process_image(message: Message, bot: Bot, file_obj, mime: str) -> None:
    buf = io.BytesIO()
    try:
        await bot.download(file_obj, destination=buf)
    except Exception as e:
        logging.exception("Не удалось скачать изображение")
        await message.answer(f"Не удалось скачать изображение: {e}"[:4000], parse_mode=None)
        return
    b64 = base64.b64encode(buf.getvalue()).decode()
    caption = (message.caption or "").strip() or "Опиши и проанализируй это изображение."
    content = [
        {"type": "text", "text": caption},
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
    ]
    await process_prompt(message, bot, content, history_text=f"[Изображение] {caption}")


@router.message(F.photo)
async def handle_photo(message: Message, bot: Bot) -> None:
    await process_image(message, bot, message.photo[-1], "image/jpeg")


@router.message(F.text)
async def handle_text(message: Message, bot: Bot) -> None:
    await process_prompt(message, bot, message.text)


@router.message(F.document)
async def handle_document(message: Message, bot: Bot) -> None:
    doc = message.document
    name = doc.file_name or "file"

    if doc.file_size and doc.file_size > MAX_FILE_BYTES:
        await message.answer(
            f"Файл слишком большой. Максимум — {MAX_FILE_BYTES // 1_000_000} МБ.",
            parse_mode=None,
        )
        return

    lower = name.lower()
    if (doc.mime_type or "").startswith("image/") or lower.endswith(IMAGE_EXTS):
        mime = doc.mime_type if (doc.mime_type or "").startswith("image/") else "image/jpeg"
        await process_image(message, bot, doc, mime)
        return

    try:
        buf = io.BytesIO()
        await bot.download(doc, destination=buf)
        data = buf.getvalue()
    except Exception as e:
        logging.exception("Не удалось скачать файл")
        await message.answer(f"Не удалось скачать файл: {e}"[:4000], parse_mode=None)
        return

    if lower.endswith(".pdf"):
        text = extract_pdf(data)
    elif lower.endswith(".docx"):
        text = extract_docx(data)
    elif lower.endswith(".pptx"):
        text = extract_pptx(data)
    elif lower.endswith(".xlsx"):
        text = extract_xlsx(data)
    else:
        text = decode_text(data)

    if not text:
        await message.answer(
            "Не получилось прочитать файл. Поддерживаются текстовые файлы "
            "(код, txt, md, json, csv и т.п.), PDF с текстом, DOCX, XLSX, PPTX и изображения.",
            parse_mode=None,
        )
        return

    note = ""
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS]
        note = f"\n[Файл обрезан до первых {MAX_FILE_CHARS} символов]"

    caption = (message.caption or "").strip() or "Проанализируй этот файл."
    content = f"{caption}\n\n--- Файл: {name} ---\n{text}{note}\n--- Конец файла ---"
    await process_prompt(message, bot, content)


@router.message()
async def handle_other(message: Message) -> None:
    await message.answer(
        "Я понимаю текст, файлы (код, txt, PDF, DOCX, XLSX, PPTX) и изображения. "
        "Голосовые, видео и стикеры не поддерживаются.",
        parse_mode=None,
    )


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
