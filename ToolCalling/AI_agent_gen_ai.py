import requests

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langgraph.prebuilt import create_react_agent
import settings

load_dotenv()

# Search Tool
search_tool = DuckDuckGoSearchRun()


# Weather Tool
@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather data for a given city.
    """
    url = (
        f"https://api.weatherstack.com/current"
        f"?access_key={settings.WEATHERSTACK_API_KEY}"
        f"&query={city}"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()

        if "current" not in data:
            return f"Could not fetch weather data. Response: {data}"

        current = data["current"]

        return (
            f"Temperature: {current['temperature']}°C\n"
            f"Weather: {', '.join(current['weather_descriptions'])}\n"
            f"Humidity: {current['humidity']}%\n"
            f"Wind Speed: {current['wind_speed']} km/h"
        )

    except Exception as e:
        return f"Error fetching weather data: {str(e)}"


# LLM
llm = ChatOpenAI(model="gpt-4.1-mini")


# ReAct Agent
agent = create_react_agent(
    model=llm,
    tools=[search_tool, get_weather_data]
)


# Invoke Agent
response = agent.invoke(
    {
        "messages": [
            (
                "user",
                "Find the capital of Madhya Pradesh, then find its current weather condition."
            )
        ]
    }
)


# Print Final Answer
for message in response["messages"]:
    print("\n", "=" * 80)
    print(type(message).__name__)
    print(message.content)