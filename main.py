import os
import asyncio
import pytz
import datetime
import aioschedule
import nest_asyncio
import pickle
import logging
import openai
import random
import tiktoken
import aiogram.exceptions
from aiogram import Bot, Dispatcher, types, enums
from url_parser import url_article_parser, get_parser_params
from aiogram.filters import Command
from aiogram.utils.chat_action import ChatActionSender
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logfile = "journal.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s")

nest_asyncio.apply()

bot = Bot(token=os.environ.get('bot_token'))
dp = Dispatcher()

openai_client = openai.AsyncOpenAI(api_key=os.environ.get('openai_token'))
admin_chat_id = int(os.environ.get('admin_chat_id'))
FactActive = int(os.environ.get('fact_job'))
reply_probability = float(os.environ.get('reply_probability'))

chat_id = 0
poll_message_id = 0
pinned_message_id = 0
total_answers = 0
opt1 = 0
opt2 = 0
opt3 = 0
PollingJob = False
JobActive = False
bot_details = None
conversations = {}
max_tokens = 4000
truncate_limit = 3500
temperature = 1
agenda = []
end_hour = 23
filename = 'saved_data.pkl'
filedata = None
chat_type = ''

class GPTSystem(StatesGroup):
  question1 = State()
  
class AgendaAdd(StatesGroup):
  question1 = State()

class AgendaDelete(StatesGroup):
  question1 = State()
  
@dp.message(Command('get_a_fact'))
async def get_a_fact(message: types.Message):
  await gpt_clear(message, True)
  content = "Расскажи неинтересный факт, начав ответ со слов 'Неинтересный факт' только для этого запроса."
  await ask_chatGPT(message, content, "user")

async def ask_chatGPT(message: types.Message, content, role):
  global conversations
  if message.chat.id not in conversations:
    conversations[message.chat.id] = []
  conversations[message.chat.id].append({"role": role, "content": content})
  await truncate_conversation(message.chat.id)
  
  max_tokens_chat = max_tokens - await get_conversation_len(message.chat.id)
  try:
    completion = await openai_client.chat.completions.create(
      model="gpt-3.5-turbo-0125",
      messages=conversations[message.chat.id],
      max_tokens=max_tokens_chat,
      temperature=temperature,
      )
  except (
      openai.APIConnectionError,
      openai.APIError,
      openai.APIResponseValidationError,
      openai.APITimeoutError,
      openai.APIResponseValidationError,
      openai.APIStatusError,
      openai.AuthenticationError,
      openai.BadRequestError,
      openai.ConflictError,
      openai.InternalServerError,      
      openai.NotFoundError,
      openai.OpenAIError,      
      openai.PermissionDeniedError,      
      openai.RateLimitError,
      openai.UnprocessableEntityError,
  ) as e:
    #print(f"\033[38;2;255;0;0mOpenAI API error: {e}\033[0m")
    logging.error(f"OpenAI API error: {e}")
    pass  
    
  gpt_finish_reason = completion.choices[0].finish_reason
  if gpt_finish_reason.lower() == 'stop':
    gpt_response = completion.choices[0].message.content
    await message.reply(gpt_response)
    conversations[message.chat.id].append({"role": "assistant", "content": gpt_response})
    await file_write()
  else: 
    text = f'❗️Ошибка OpenAI API: {gpt_finish_reason}'
    await message.answer(text)

async def truncate_conversation(chat_id: int):
  global conversations
  global truncate_limit
  while True:
    conversation_len = await get_conversation_len(chat_id)
    if conversation_len > truncate_limit and chat_id in conversations:
      now = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
      #print(f"\033[38;2;128;0;128m{now.strftime('%d.%m.%Y %H:%M:%S')} | Conversation size is {conversation_len} tokens, thus it will be truncated\033[0m")
      logging.info(f"Conversation size is {conversation_len} tokens, thus it will be truncated")
      conversations[chat_id].pop(0) 
    else:
      break

async def get_conversation_len(chat_id: int) -> int:
  global conversations
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  num_tokens = 0
  for msg in conversations[chat_id]:
    # every message follows <im_start>{role/name}\n{content}<im_end>\n
    num_tokens += 5
    for key, value in msg.items():
      num_tokens += len(encoding.encode(value))
      if key == "name":  # if there's a name, the role is omitted
          num_tokens += 5  # role is always required and always 1 token
  num_tokens += 5  # every reply is primed with <im_start>assistant
  return num_tokens

async def get_prompt_len(prompt: dict) -> int:
  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  num_tokens = 0
  # every message follows <im_start>{role/name}\n{content}<im_end>\n
  num_tokens += 5
  for msg in prompt:
    for key, value in msg.items():
      num_tokens += len(encoding.encode(value))
      if key == "name":  # if there's a name, the role is omitted
        num_tokens += 5  # role is always required and always 1 token
  return num_tokens

@dp.message(Command('gpt_system'))
async def gpt_system_message(message: types.Message, state: FSMContext):
  await state.set_state(GPTSystem.question1)
  await message.answer("Введите системное сообщение...")

@dp.message(GPTSystem.question1)
async def gpt_system_question1_handler(message: types.Message, state: FSMContext):
  await state.clear()
  await default_message_handler(message, "system")
  
@dp.message(Command('gpt_clear'))
async def gpt_clear(message: types.Message, silent_mode=False):
  global conversations
  if message.chat.id in conversations:
    del conversations[message.chat.id]
    await file_write()

  if not silent_mode:
    text = '❗️История переписки с ChatGPT очищена 💪'
    await message.answer(text, parse_mode="HTML")
    
@dp.message(Command('gpt_clear_all'))
async def gpt_clear_all(message: types.Message=None):
  global conversations
  conversations = {}
  await file_write()
  now = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
  #print(f"\033[38;2;128;0;128m{now.strftime('%d.%m.%Y %H:%M:%S')} | Job 'gpt_clear_all' is completed\033[0m")
  logging.info("Job 'gpt_clear_all' is completed")

@dp.message(Command('gpt_show'))
async def gpt_show(message: types.Message):
  global conversations
  if message.chat.id in conversations:
    message = await bot.send_message(message.chat.id, f"Chat_id: {message.chat.id}")
    LastMessage_id = message.message_id
    LastMessage_text = message.text + "\n"
    for msg in conversations[message.chat.id]:
      LastMessage_text += f"- {msg['content']}\n"
      await bot.edit_message_text(chat_id=message.chat.id, message_id=LastMessage_id, text=LastMessage_text)
  else:
    text = '❗️История переписки с ChatGPT пустая'
    await message.answer(text, parse_mode="HTML")

@dp.message(Command('gpt_show_all'))
async def gpt_show_all(message: types.Message):
  global conversations
  if conversations:
    for chat_id in conversations:
      message = await bot.send_message(message.chat.id, f"Chat_id: {chat_id}")
      LastMessage_id = message.message_id
      LastMessage_text = message.text + "\n"
      for msg in conversations[chat_id]:
        LastMessage_text += f"- {msg['content']}\n"
        await bot.edit_message_text(chat_id=message.chat.id, message_id=LastMessage_id, text=LastMessage_text)
  else:
    text = '❗️История переписки с ChatGPT пустая'
    await message.answer(text, parse_mode="HTML")

@dp.message(Command('agenda_add'))
async def agenda_add(message: types.Message, state: FSMContext):
  error_code = await check_authority(message, 'agenda_add')
  if error_code != 0:
    return
  
  await state.set_state(AgendaAdd.question1) 
  await message.answer("С новой строки введите один или несколько пунктов для добавления в повестку заседания Клуба 101 или введите 'Нет' для прерывания команды...")
  
@dp.message(AgendaAdd.question1)
async def agenda_add_question1_handler(message: types.Message, state: FSMContext):
  await state.clear()
  global agenda
  if message.text.lower() != "нет":
    lines = message.text.splitlines()
    for line in lines:
      agenda.append(line)
    await file_write()
  else:
    text = '❗️Команда сброшена 🤬'
    await message.answer(text, parse_mode="HTML")
  await agenda_show(message)

@dp.message(Command('agenda_delete'))
async def agenda_delete(message: types.Message, state: FSMContext):
  error_code = await check_authority(message, 'agenda_delete')
  if error_code != 0:
    return
      
  global agenda
  if agenda != []:
    await agenda_show(message)
    await state.set_state(AgendaDelete.question1)
    await message.answer("Введите через запятую номера позиций, которые нужно удалить из повестки заседания Клуба 101 или введите 'Нет' для прерывания команды...")
  else:
    text = '❗️Повестка заседания Клуба 101 пустая - удалять нечего 😢'
    await message.answer(text, parse_mode="HTML")

@dp.message(AgendaDelete.question1)
async def agenda_delete_question1_handler(message: types.Message, state: FSMContext):
  global agenda
  temp_agenda = []
  num_array = []
  await state.clear()
  if message.text.lower() != "нет":
    lines_to_delete = message.text.split(',')
    for line_num in lines_to_delete:
      try:
        num = int(line_num.strip())
        if num > 0 and num <= len(agenda):
          num_array.append(num)
      except ValueError:
        #ignore non-integer line numbers
        pass
    if num_array != []:
      for i, line in enumerate(agenda):
        if i+1 not in num_array:
          temp_agenda.append(line)
      agenda = temp_agenda
      await file_write()
  else:
    text = '❗️Команда сброшена 🤬'
    await message.answer(text, parse_mode="HTML")
  await agenda_show(message)

@dp.message(Command('agenda_show'))
async def agenda_show(message: types.Message):
  error_code = await check_authority(message, 'agenda_show')
  if error_code != 0:
    return
      
  global agenda
  mes = []
  if agenda != []:
    text = '❗️Повестка заседания Клуба 101:'
    mes.append(text)
    for i, item in enumerate(agenda):
      item_id = i + 1
      text = f'👉 {item_id}. {item}'
      mes.append(text)
    await bot.send_message(message.chat.id,'\n'.join(mes), parse_mode="HTML")
  else:
    text = '❗️Повестка заседания Клуба 101 пока пустая 😢'
    await bot.send_message(message.chat.id, text, parse_mode="HTML")
  
@dp.message(Command('agenda_clear'))
async def agenda_clear(message: types.Message):
  error_code = await check_authority(message, 'agenda_clear')
  if error_code != 0:
    return
      
  global agenda
  agenda = []
  await file_write()
  text = '❗️Повестка заседания Клуба 101 очищена 💪'
  await message.answer(text, parse_mode="HTML")

@dp.message(lambda message: not message.text.startswith('/'))
async def default_message_handler(message: types.Message, role: str="user"):
  article_text = []
  url_yes = False
  parser_option = 1
  orig_url = False
  if message.chat.type != enums.chat_type.ChatType.PRIVATE and role != "system":
    if f'@{bot_details.username}' in message.text:
      content = message.text.replace(f'@{bot_details.username}', '').strip()
    elif message.reply_to_message and message.reply_to_message.from_user.username == bot_details.username:
      content = message.text
    elif random.random() <= reply_probability:
      responses = [
          "Усомнись в данном сообщении\n",
          "Вырази случайное мнение по поводу данного сообщения\n",
          "Задайся вопросом по поводу данного сообщения, ответ на который не содержится в самом сообщении\n",
          "Сформулируй критическое мнение по данному сообщению\n",
          "Расскажи что-то еще интересное по теме из данного сообщения\n",
          "Расскажи забавный факт по теме из данного сообщения\n",
      ]
      content = random.choice(responses)
      await gpt_clear(message, True)
      if message.text:
        content += message.text
      if message.caption:
        content += message.caption
      if message.reply_to_message:
        if message.reply_to_message.text:
          content += message.reply_to_message.text
        if message.reply_to_message.caption:
          content += message.reply_to_message.caption

      async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        await ask_chatGPT(message, content, "user")
        return
    else:
      return
  else:
    content = message.text
  
  if message.entities is not None:
    for entity in message.entities:
      if entity.type == "url":
        url = message.text[entity.offset: entity.offset + entity.length]
        if url.startswith('http'):
          params = await get_parser_params(message.text)
          parser_option = params['parser_option']
          orig_url = params['orig_url']
          article_text = await url_article_parser(url=url, parser_option=parser_option, orig_url=orig_url)
          content = content.replace(f'parser_option{parser_option}', '').strip()
          content = content.replace('orig_url', '').strip()
          if article_text != '':
            content = content.replace(url, '')
            content += "\n" + article_text
  
  if message.reply_to_message:
    if message.reply_to_message.entities is not None:
      for entity in message.reply_to_message.entities:
        if entity.type == "url":
          url = message.reply_to_message.text[entity.offset: entity.offset + entity.length]
          if url.startswith('http'):
            params = await get_parser_params(message.text)
            parser_option = params['parser_option']
            orig_url = params['orig_url']
            article_text = await url_article_parser(url=url, parser_option=parser_option, orig_url=orig_url)
            content = content.replace(f'parser_option{parser_option}', '').strip()
            content = content.replace('orig_url', '').strip()
            if article_text != '':
              url_yes = True
              content += "\n" + article_text
              break
    
    if not url_yes:
      if message.reply_to_message.text:
        reply_to_text = message.reply_to_message.text
        if bot_details.username in reply_to_text:
          reply_to_text = reply_to_text.replace(f'@{bot_details.username}', '').strip()
        if reply_to_text:
          content += "\n" + reply_to_text
      elif message.reply_to_message.caption:
        content += "\n" + message.reply_to_message.caption

  prompt_len = await get_prompt_len(prompt=[{"role": role, "content": content}])
  if prompt_len > max_tokens:
    text = f'❗️Длина запроса {prompt_len} токенов > максимальной длины разговора {max_tokens}'
    await message.answer(text, parse_mode="HTML")
    return
    
  async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
    await ask_chatGPT(message, content, role)

@dp.message(Command('send_poll_now'))
async def send_poll(message: types.Message):
  error_code = await check_authority(message, 'send_poll_now')
  if error_code != 0:
    return
    
  if message.chat.type == enums.chat_type.ChatType.PRIVATE:
    text = '❗️Запуск голосования возможен только из группового чата'
    await bot.send_message(message.chat.id, text, parse_mode="HTML")
    return
      
  global chat_id
  global chat_type
  global poll_message_id
  global total_answers
  global opt1
  global opt2
  global opt3
  total_answers = 0
  opt1 = 0
  opt2 = 0
  opt3 = 0  
  if chat_id == 0 and message.chat.id != 0:
    chat_id = message.chat.id
    chat_type = message.chat.type
  await unpin_poll_results(silent_mode=True)
  moscow_tz = pytz.timezone('Europe/Moscow')
  now = datetime.datetime.now(moscow_tz)
  days_until_friday = (4 - now.weekday()) % 7
  fri_date = now.date() + datetime.timedelta(days=days_until_friday)
  option1 = 'Пятница (' + fri_date.strftime('%d.%m.%Y') + ')'
  days_until_saturday = (5 - now.weekday()) % 7
  sat_date = now.date() + datetime.timedelta(days=days_until_saturday)
  option2 = 'Суббота (' + sat_date.strftime('%d.%m.%Y') + ')'
  days_until_sunday = (6 - now.weekday()) % 7
  sun_date = now.date() + datetime.timedelta(days=days_until_sunday)
  option3 = 'Воскресенье (' + sun_date.strftime('%d.%m.%Y') + ')'
  option4 = 'Пропущу в этот раз 😢'
  options = [option1, option2, option3, option4]
  poll_question = 'Когда состоится следующее заседание Клуба 101? 😤'
  poll_message = await bot.send_poll(chat_id, poll_question, options=options, is_anonymous=False, allows_multiple_answers=True)
  poll_message_id = poll_message.message_id
  await file_write()
  await bot.pin_chat_message(chat_id=chat_id, message_id=poll_message_id)
  text = f'❗️Голосование длится до {end_hour}:00 или до получения 4 голосов за один из вариантов кроме последнего.\nМожно выбрать несколько вариантов ответа.\nДля подтверждения ввода обязательно нажать <b>VOTE</b>.'
  await message.answer(text, parse_mode="HTML")
  await wait_for_poll_stop()

async def wait_for_poll_stop():
    moscow_tz = pytz.timezone('Europe/Moscow')
    now = datetime.datetime.now(moscow_tz)
    target_time = now.replace(hour=end_hour, minute=0, second=0, microsecond=0)
    time_until_poll_stop = (target_time - now).total_seconds()   
    if time_until_poll_stop> 0:
      await asyncio.sleep(time_until_poll_stop)
    try:
      await bot.stop_poll(chat_id, poll_message_id)
    except aiogram.exceptions.DetailedAiogramError("Poll Has Already Been Closed"):
      pass

@dp.poll(lambda closed_poll: closed_poll.is_closed is True)
async def poll_results(closed_poll: types.Poll):
  global pinned_message_id
  global poll_message_id
  message = types.Message(chat=types.Chat(id=chat_id,type=chat_type),date=datetime.datetime.now(),message_id=0)
  max_option_1 = closed_poll.options[0].text
  max_votes_1 = closed_poll.options[0].voter_count
  max_id_1 = 1
  max_option_2 = closed_poll.options[0].text
  max_votes_2 = closed_poll.options[0].voter_count
  max_id_2 = 1
  if poll_message_id != 0:
    await bot.unpin_chat_message(chat_id=chat_id, message_id=poll_message_id)
    poll_message_id = 0
    await file_write()
  for i, option in enumerate(closed_poll.options):
    if option.voter_count >= max_votes_1:
      max_option_2 = max_option_1
      max_votes_2 = max_votes_1
      max_id_2 = max_id_1
      max_option_1 = option.text
      max_votes_1 = option.voter_count
      max_id_1 = i+1
  if max_votes_1 < 2:
    text = '❗️Голосование завершено. Решение не принято - слишком мало голосов 🤬'
    message = await bot.send_message(chat_id, text, parse_mode="HTML")
  else:
    if max_id_1 == 4:
      if max_id_2 != max_id_1 and max_votes_2 > 1:
        text = f'❗️Голосование завершено. Большинство решили слиться, но заседание на этой неделе все же состоится в <b>{max_option_2}</b> 👍'
        message = await bot.send_message(chat_id, text, parse_mode="HTML")
        pinned_message_id = message.message_id
        await bot.pin_chat_message(chat_id=chat_id, message_id=pinned_message_id)
        await file_write()
        await agenda_show(message)
      else:
        text = '❗️Голосование завершено. Заседание на этой неделе не состоится - большинство не может участвовать 👎'
        await bot.send_message(chat_id, text, parse_mode="HTML")
    else:
      if max_votes_1 == max_votes_2 and max_id_1 != max_id_2:
        text = f'❗️Голосование завершено. Заседание на этой неделе состоится в <b>{max_option_2}</b> и в <b>{max_option_1}</b>👍'
      else:
        text = f'❗️Голосование завершено. Заседание на этой неделе состоится в <b>{max_option_1}</b> 👍'
      message = await bot.send_message(chat_id, text, parse_mode="HTML")
      pinned_message_id = message.message_id
      await bot.pin_chat_message(chat_id=chat_id, message_id=pinned_message_id)
      await file_write()
      await agenda_show(message)
 
@dp.poll_answer(lambda poll_answer: True)
async def poll_answer(poll_answer: types.PollAnswer):
  global total_answers
  global opt1
  global opt2
  global opt3
  total_answers += 1
  if total_answers == 9:
    await bot.stop_poll(chat_id, poll_message_id)
    total_answers = 0
    opt1 = 0      
    opt2 = 0
    opt3 = 0
  else:
    for i, opt_id in enumerate(poll_answer.option_ids):
      if opt_id == 0:
        opt1 += 1
      elif opt_id == 1:
        opt2 += 1
      elif opt_id == 2:
        opt3 += 1
    if opt1 == 4 or opt2 == 4 or opt3 == 4:
      await bot.stop_poll(chat_id, poll_message_id)
      total_answers = 0
      opt1 = 0      
      opt2 = 0
      opt3 = 0

async def unpin_poll_results(silent_mode=False):
  global pinned_message_id
  if pinned_message_id != 0:
    await bot.unpin_chat_message(chat_id=chat_id, message_id=pinned_message_id)
    pinned_message_id = 0
    await file_write()
  now = datetime.datetime.now(pytz.timezone('Europe/Moscow'))
  if not silent_mode:
    #print(f"\033[38;2;128;0;128m{now.strftime('%d.%m.%Y %H:%M:%S')} | Job 'unpin_poll_results' is completed\033[0m")
    logging.info("Job 'unpin_poll_results' is completed")

async def polling_reminder():
  if poll_message_id != 0:
    text = '🫵 А ты уже проголосовал за дату следующего заседания Клуба 101? Ссылка в закрепе 👆'
    await bot.send_message(chat_id, text, parse_mode="HTML")
  
async def polling_job(message: types.Message, silent_mode=False):
  aioschedule.every().thursday.at('09:00').do(send_poll, message=message)
  aioschedule.every().thursday.at('17:00').do(polling_reminder) 
  
  if not silent_mode:
    text = '❗️Опрос по расписанию запущен 💪'
    moscow_tz = pytz.timezone('Europe/Moscow')
    utc_time = aioschedule.jobs[0].next_run
    moscow_time = utc_time.astimezone(moscow_tz)
    time_str = moscow_time.strftime('%Y-%m-%d %H:%M:%S')
    text += f'\nСледующий опрос состоится <b>{time_str} MSK</b>'
    await bot.send_message(chat_id, text, parse_mode="HTML")

async def fact_job(message: types.Message):
  aioschedule.every().day.at('13:00').do(get_a_fact, message=message)

async def maintenance_job():
  aioschedule.every().day.at('22:00').do(gpt_clear_all)
  aioschedule.every().sunday.at('22:01').do(unpin_poll_results)
  aioschedule.every().sunday.at('22:02').do(clear_logfile)

@dp.message(Command('schedule_start'))
async def schedule_start(message: types.Message):
  global PollingJob
  global JobActive
  global chat_id
  global chat_type
  
  if message.chat.type != enums.chat_type.ChatType.PRIVATE:
    if not chat_id:
      chat_id = message.chat.id
      chat_type = message.chat.type
    error_code = await check_authority(message, 'schedule_start')
    if error_code != 0:
      return
    PollingJob = True
    JobActive = True
    await file_write()
    await schedule_jobs(message)
  else:
    text = '❗️Запуск плановых задач возможен только из группового чата'
    await bot.send_message(message.chat.id, text, parse_mode="HTML")

@dp.message(Command('schedule_check'))
async def schedule_check(message: types.Message):
  error_code = await check_authority(message, 'schedule_check')
  if error_code != 0:
    return

  if PollingJob:
    text = '❗️Опрос по расписанию активен'
    moscow_tz = pytz.timezone('Europe/Moscow')
    utc_time = aioschedule.jobs[0].next_run  #Опрос всегда первый в списке для упрощения
    moscow_time = utc_time.astimezone(moscow_tz)
    time_str = moscow_time.strftime('%Y-%m-%d %H:%M:%S')
    text += f'\nСледующий опрос состоится <b>{time_str} MSK</b>'
  else:
    text = '❗️Опрос по расписанию неактивен'
  if JobActive:
    text += '\n❗️Плановые задачи выполняются'
  else:
    text += '\n❗️Плановые задачи остановлены'
  await message.answer(text, parse_mode="HTML")
    
@dp.message(Command('schedule_stop'))
async def schedule_stop(message: types.Message):
  error_code = await check_authority(message, 'schedule_stop')
  if error_code != 0:
    return
      
  global PollingJob
  PollingJob = False
  await file_write()
  await schedule_jobs(message)    
  text = '❗️Опрос по расписанию остановлен 💪'
  await message.answer(text, parse_mode="HTML")

@dp.message(Command('clear_log'))
async def clear_logfile(message: types.Message=None, Job=False):
  if Job:
    max_size_bytes = 1024 * 1024
  else:
    max_size_bytes = 0

  if not os.path.exists(logfile):
    raise FileNotFoundError(f"Log file '{logfile}' does not exist.")

  # Get file size
  file_size = os.path.getsize(logfile)

  if file_size > max_size_bytes:
    logging.info(f"Log file '{logfile}' exceeded size limit ({max_size_bytes} bytes). Cleaning...")
    try:
      with open(logfile, "w") as f:
        # Clear the file content
        f.write("")
      logging.info(f"Log file '{logfile}' cleaned successfully.")
    except Exception as e:
      logging.error(f"Error cleaning log file: {e}")
  else:
    logging.info(f"Log file has size of {file_size} bytes")

async def schedule_jobs(message: types.Message, silent_mode=False):
  aioschedule.clear()
  if PollingJob:
    asyncio.create_task(polling_job(message, silent_mode))
  if JobActive:
    asyncio.create_task(maintenance_job())
  if FactActive == 1:
    asyncio.create_task(fact_job(message))
  if poll_message_id != 0:
    asyncio.create_task(wait_for_poll_stop())

async def check_authority(message, command):
  error_code = 0
  commands = ['gpt_clear']
  if command not in commands:
    if (message.chat.type == enums.chat_type.ChatType.PRIVATE and message.from_user.id != admin_chat_id) or (message.chat.type != enums.chat_type.ChatType.PRIVATE and message.chat.id != chat_id):
      text = "❗️Нет доступа к команде"
      await bot.send_message(message.chat.id, text)
      error_code = 4
  return error_code
  
async def file_read():
  global JobActive
  global PollingJob
  global pinned_message_id
  global poll_message_id
  global total_answers
  global opt1
  global opt2
  global opt3
  global chat_id
  global chat_type
  global agenda
  global conversations

  if os.path.exists(filename) and os.path.getsize(filename) > 0:
    with open(filename, 'rb') as f:
      filedata = pickle.load(f)
    JobActive = filedata["JobActive"]
    PollingJob = filedata["PollingJob"]
    pinned_message_id = filedata["pinned_message_id"]
    poll_message_id = filedata["poll_message_id"]
    total_answers = filedata["total_answers"]
    opt1 = filedata["opt1"]
    opt2 = filedata["opt2"]
    opt3 = filedata["opt3"]
    chat_id = filedata["chat_id"]
    chat_type = filedata["chat_type"]
    agenda = filedata["agenda"]
    conversations = filedata["conversations"]

async def file_write():
  if os.path.exists(filename):
    filedata = {"JobActive": JobActive,
                "PollingJob": PollingJob,
                "pinned_message_id": pinned_message_id,
                "poll_message_id": poll_message_id,
                "total_answers": total_answers,
                "opt1": opt1,
                "opt2": opt2,
                "opt3": opt3,
                "chat_id": chat_id,
                "chat_type": chat_type,
                "agenda": agenda,
                "conversations": conversations}
    with open(filename, 'wb') as f:
      pickle.dump(filedata, f)

async def file_init():
  if os.path.exists(filename) and os.path.getsize(filename) == 0:
    filedata = {"JobActive": False,
                "PollingJob": False,
                "pinned_message_id": 0,
                "poll_message_id": 0,
                "total_answers": 0,
                "opt1": 0,
                "opt2": 0,
                "opt3": 0,
                "chat_id": 0,
                "chat_type": "",
                "agenda": [],
                "conversations": {}}
    with open(filename, 'wb') as f:
      pickle.dump(filedata, f)

async def run_scheduled_jobs():
  while True:
    await aioschedule.run_pending()
    await asyncio.sleep(1)


async def main():
  global bot_details
  bot_details = await bot.get_me()
  await file_init()
  await file_read()
  if JobActive:
    message = types.Message(chat=types.Chat(id=chat_id,type=chat_type),date=datetime.datetime.now(),message_id=0)
    await schedule_jobs(message, silent_mode=True)
  job_loop = asyncio.get_event_loop()
  job_loop.create_task(run_scheduled_jobs())
  await bot.delete_webhook(drop_pending_updates=True)
  await dp.start_polling(bot)

if __name__ == '__main__':
  # keep_alive()
  main_loop = asyncio.get_event_loop() 
  main_loop.run_until_complete(main())