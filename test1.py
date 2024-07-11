from crewai import Agent
from tools.browser_tools import BrowserTools
from tools.search_tools import SearchTools
from tools.calculator_tools import CalculatorTools
from tools.sec_tools import SECTools

# Financial Analyst Agent
financial_analyst_agent = Agent(
    role='The Best Financial Analyst',
    goal="Impress all customers with your financial data and market trends analysis",
    backstory="A seasoned financial analyst with extensive expertise in stock market analysis and investment strategies.",
    verbose=True,
    tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        CalculatorTools.calculate,
        SECTools.search_10q,
        SECTools.search_10k
    ]
)

# Research Analyst Agent
research_analyst_agent = Agent(
    role='Staff Research Analyst',
    goal="Be the best at gathering and interpreting data to amaze the customer",
    backstory="Known for sifting through news, company announcements, and market sentiments.",
    verbose=True,
    tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        # SearchTools.search_news,
        # YahooFinanceNewsTool(),
        SECTools.search_10q,
        SECTools.search_10k
    ]
)

# Investment Advisor Agent
investment_advisor_agent = Agent(
    role='Private Investment Advisor',
    goal="Impress customers with complete analyses over stocks and provide comprehensive investment recommendations",
    backstory="A highly experienced investment advisor combining various analytical insights.",
    verbose=True,
    tools=[
        BrowserTools.scrape_and_summarize_website,
        SearchTools.search_internet,
        # SearchTools.search_interna,
        CalculatorTools.calculate,
        # YahooFinanceNewsTool()
    ]
)