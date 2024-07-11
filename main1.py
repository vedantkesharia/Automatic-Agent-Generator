import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import AzureChatOpenAI
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
# Loading Tools
load_dotenv()

# os.environ["OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

search_tool = SerperDevTool()

llmhey = AzureChatOpenAI(
    openai_api_version="2024-05-01-preview",
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# llmhey = ChatGroq(api_key=os.getenv("GROQ_API_KEY"),  # Replace with the appropriate API key parameter
#             model_name="llama3-70b-8192",)

# Define your agents with roles, goals, tools, and additional attributes
researcher = Agent(
    # model = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    role='Senior Research Analyst',
    goal='Uncover cutting-edge developments in AI and data science',
    backstory=(
        "You are a Senior Research Analyst at a leading tech think tank. "
        "Your expertise lies in identifying emerging trends and technologies in AI and data science. "
        "You have a knack for dissecting complex data and presenting actionable insights."
    ),
    verbose=True,
    allow_delegation=False,
    tools=[search_tool],
    llm = llmhey,
    # api_key=os.getenv("AZURE_OPENAI_API_KEY")
)
writer = Agent(
    # model = os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    role='Tech Content Strategist',
    goal='Craft compelling content on tech advancements',
    backstory=(
        "You are a renowned Tech Content Strategist, known for your insightful and engaging articles on technology and innovation. "
        "With a deep understanding of the tech industry, you transform complex concepts into compelling narratives."
    ),
    verbose=True,
    allow_delegation=True,
    tools=[search_tool],
    cache=False,  # Disable cache for this agent
    llm = llmhey,
    # api_key=os.getenv("AZURE_OPENAI_API_KEY")
)

# Create tasks for your agents
task1 = Task(
    description=(
        "Conduct a comprehensive analysis of the latest advancements in AI in 2024. "
        "Identify key trends, breakthrough technologies, and potential industry impacts. "
        "Compile your findings in a detailed report. "
        "Make sure to check with a human if the draft is good before finalizing your answer."
    ),
    expected_output='A comprehensive full report on the latest AI advancements in 2024, leave nothing out',
    agent=researcher,
    human_input=True
)

task2 = Task(
    description=(
        "Using the insights from the researcher\'s report, develop an engaging blog post that highlights the most significant AI advancements. "
        "Your post should be informative yet accessible, catering to a tech-savvy audience. "
        "Aim for a narrative that captures the essence of these breakthroughs and their implications for the future."
    ),
    expected_output='A compelling 3 paragraphs blog post formatted as markdown about the latest AI advancements in 2024',
    agent=writer
)

# Instantiate your crew with a sequential process
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    verbose=2,
    memory=True,
)

# Get your crew to work!
result = crew.kickoff()

print("######################")
print(result)