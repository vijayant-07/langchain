import requests
import json

from langchain_core.tools import tool, InjectedToolArg
from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
  """
  This function fetches the currency conversion factor between a given base currency and a target currency
  """
  url = f'https://v6.exchangerate-api.com/v6/b18ad8d3183d85e6c7316a37/pair/{base_currency}/{target_currency}'

  response = requests.get(url)

  return response.json()

@tool
def convert(base_currency_value: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
  """
  given a currency conversion rate this function calculates the target currency value from a given base currency value
  """

  return base_currency_value * conversion_rate


get_conversion_factor.invoke({'base_currency':'USD','target_currency':'INR'})

convert.invoke({'base_currency_value':10, 'conversion_rate':85.16})

llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

messages = [HumanMessage('What is the conversion factor between INR and USD, and based on that can you convert 10 usd to inr')]

print(messages)

ai_message = llm_with_tools.invoke(messages)

messages.append(ai_message)

ai_message.tool_calls


for tool_call in ai_message.tool_calls:
  # execute the 1st tool and get the value of conversion rate
  if tool_call['name'] == 'get_conversion_factor':
    tool_message1 = get_conversion_factor.invoke(tool_call)
    # fetch this conversion rate
    conversion_rate = json.loads(tool_message1.content)['conversion_rate']
    # append this tool message to messages list
    messages.append(tool_message1)
  # execute the 2nd tool using the conversion rate from tool 1
  if tool_call['name'] == 'convert':
    # fetch the current arg
    tool_call['args']['conversion_rate'] = conversion_rate
    tool_message2 = convert.invoke(tool_call)
    messages.append(tool_message2)


print(llm_with_tools.invoke(messages).content)