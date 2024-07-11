import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import AzureChatOpenAI
from crewai_tools import CodeDocsSearchTool,GithubSearchTool,WebsiteSearchTool
from langchain_groq.chat_models import ChatGroq
load_dotenv()

serper = os.getenv('SERPER_API_KEY')

class CodeGenerationAgents():
    def __init__(self):
        self.llm = AzureChatOpenAI(
            openai_api_version="2024-05-01-preview",
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        # self.llm = ChatGroq(
        #     api_key=os.getenv("GROQ_API_KEY"),  # Replace with the appropriate API key parameter
        #     model_name="llama3-70b-8192",
        # )
        # self.website_search_tool = WebsiteSearchTool(website='https://www.crewai.com/')
        self.docs_search_tool = CodeDocsSearchTool(docs_url='https://docs.crewai.com/',config=dict(
                llm=dict(
                    provider="openai", 
                    # provider="groq",
                    config=dict(
                        # model="llama3-70b-8192",
                        model="Gpt4o",
                    ),
                ),
                embedder=dict(
                    provider="google",
                    config=dict(
                        model="models/embedding-001",
                        # model="llama3-70b-8192",
                        task_type="retrieval_document",
                    ),
                ),
            ))
        self.github_search_tool = GithubSearchTool(github_repo='vedantkesharia/crewAI-examples',content_types=['code','repo'],  gh_token=os.getenv("GITHUB_TOKEN"), config=dict(
                # llm=dict(
                #     provider="groq", 
                #     config=dict(
                #         model="llama3-70b-8192",
                #     ),
                # ),
                llm=dict(
                    provider="openai", 
                    # provider="groq",
                    config=dict(
                        model="Gpt4o",
                    ),
                ),
                embedder=dict(
                    provider="google", 
                    # provider="groq", 
                    config=dict(
                        model="models/embedding-001",
                        # model="llama3-70b-8192",
                        task_type="retrieval_document",
                    ),
                ),
            ))

    def code_generator(self):
        return Agent(
            # model=os.getenv("MODEL"),
            role='Code Generator',
            goal='Generate CrewAI agents based on user input using CrewAI documentation.',
            backstory='You are a proficient code generator with extensive knowledge of the CrewAI framework.',
            tools=[
                self.docs_search_tool,
                self.github_search_tool
            ],
            llm=self.llm,
            verbose=True,
            max_iter=15
        )

    def code_evaluator(self):
        return Agent(
            # model=os.getenv("MODEL"),
            role='Code Evaluator',
            goal='Evaluate the generated CrewAI agents and provide feedback using CrewAI documentation.',
            backstory='You are an experienced code reviewer with an eye for detail.',
            tools=[
                self.docs_search_tool,
                self.github_search_tool
            ],
            llm=self.llm,
            verbose=True,
            max_iter=15
        )

    def code_reviser(self):
        return Agent(
            # model=os.getenv("MODEL"),
            role='Code Reviser',
            goal='Revise the CrewAI agents based on feedback from the Code Evaluator using CrewAI documentation.',
            backstory='You are adept at refining code to perfection based on evaluations.',
            tools=[
                self.docs_search_tool,
                self.github_search_tool
            ],
            llm=self.llm,
            verbose=True,
            max_iter=15
        )

def main():
    agents = CodeGenerationAgents()

    # Runtime user input
    user_input = input("Please enter the type of agent you want to create: ")

    # Create the agents
    code_generator_agent = agents.code_generator()
    code_evaluator_agent = agents.code_evaluator()
    code_reviser_agent = agents.code_reviser()

    # Tasks
    generate_agent_task = Task(
        description='Generate CrewAI agents based on the following user input: {user_input}.',
        expected_output='CrewAI agents definition in Python format.',
        agent=code_generator_agent,
        human_input=True
    )

    evaluate_agent_task = Task(
        description='Evaluate the following generated CrewAI agents: {generated_agent}. Provide detailed feedback on its correctness and quality.',
        expected_output='Detailed feedback on the generated CrewAI agents.',
        agent=code_evaluator_agent,
        human_input=True
    )

    revise_agent_task = Task(
        description='Revise the following CrewAI agents based on the feedback: {feedback}.',
        expected_output='Revised CrewAI agents in Python format.',
        agent=code_reviser_agent,
        human_input=True
    )

    # Forming the crew
    crew = Crew(
        agents=[code_generator_agent, code_evaluator_agent, code_reviser_agent],
        tasks=[generate_agent_task],
        process=Process.sequential,
        memory=True
    )

    # Kickoff the crew with dynamic user input for agent generation
    generate_agent_result = crew.kickoff(inputs={'user_input': user_input})
    
    # Debugging: Print the result structure
    print("Generate Agent Result Structure:", generate_agent_result)

    generated_agent = generate_agent_result

    print("Generated Agent:")
    print(generated_agent)

    # Update the tasks and inputs for evaluation and revision
    crew.tasks = [evaluate_agent_task]
    evaluate_agent_result = crew.kickoff(inputs={'generated_agent': generated_agent})
    
    # Debugging: Print the result structure
    print("Evaluate Agent Result Structure:", evaluate_agent_result)

    feedback = evaluate_agent_result

    print("Feedback:")
    print(feedback)

    crew.tasks = [revise_agent_task]
    revise_agent_result = crew.kickoff(inputs={'feedback': feedback})
    
    # Debugging: Print the result structure
    print("Revise Agent Result Structure:", revise_agent_result)

    revised_agent = revise_agent_result

    print("Revised Agent:")
    print(revised_agent)

if __name__ == "__main__":
    main()








# import os
# from dotenv import load_dotenv
# from crewai import Agent, Task, Crew, Process
# from tools.calculator_tools import CalculatorTools
# from tools.search_tools import SearchTools
# from langchain_openai import AzureChatOpenAI

# load_dotenv()

# class CodeGenerationAgents():
#     def __init__(self):
#         self.llm = AzureChatOpenAI(
#             openai_api_version="2024-05-01-preview",
#             azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY")
#         )

#     def code_generator(self):
#         return Agent(
#             role='Code Generator',
#             goal='Generate agent code based on user input.',
#             backstory='You are a proficient code generator with extensive knowledge of CrewAI framework.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

#     def code_evaluator(self):
#         return Agent(
#             role='Code Evaluator',
#             goal='Evaluate the generated code and provide feedback.',
#             backstory='You are an experienced code reviewer with an eye for detail.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

#     def code_reviser(self):
#         return Agent(
#             role='Code Reviser',
#             goal='Revise the code based on feedback from the Code Evaluator.',
#             backstory='You are adept at refining code to perfection based on evaluations.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

# def main():
#     agents = CodeGenerationAgents()

#     # Runtime user input
#     user_input = input("Please enter the type of agent you want to create: ")

#     # Create the agents
#     code_generator_agent = agents.code_generator()
#     code_evaluator_agent = agents.code_evaluator()
#     code_reviser_agent = agents.code_reviser()

#     # Tasks
#     generate_code_task = Task(
#         description='Generate agent code based on the following user input: {user_input}.',
#         expected_output='The agent code in Python format.',
#         agent=code_generator_agent,
#     )

#     evaluate_code_task = Task(
#         description='Evaluate the following generated code: {generated_code}. Provide detailed feedback on its correctness and quality.',
#         expected_output='Detailed feedback on the generated code.',
#         agent=code_evaluator_agent,
#     )

#     revise_code_task = Task(
#         description='Revise the following code based on the feedback: {feedback}.',
#         expected_output='Revised code in Python format.',
#         agent=code_reviser_agent,
#     )

#     # Forming the crew
#     crew = Crew(
#         agents=[code_generator_agent, code_evaluator_agent, code_reviser_agent],
#         tasks=[generate_code_task],
#         process=Process.sequential
#     )

#     # Kickoff the crew with dynamic user input for code generation
#     generate_code_result = crew.kickoff(inputs={'user_input': user_input})
    
#     # Debugging: Print the result structure
#     print("Generate Code Result Structure:", generate_code_result)

#     # Assuming the output format to be explored
#     generated_code = generate_code_result

#     print("Generated Code:")
#     print(generated_code)

#     # Update the tasks and inputs for evaluation and revision
#     crew.tasks = [evaluate_code_task]
#     evaluate_code_result = crew.kickoff(inputs={'generated_code': generated_code})
    
#     # Debugging: Print the result structure
#     print("Evaluate Code Result Structure:", evaluate_code_result)

#     feedback = evaluate_code_result['result']

#     print("Feedback:")
#     print(feedback)

#     crew.tasks = [revise_code_task]
#     revise_code_result = crew.kickoff(inputs={'feedback': feedback})
    
#     # Debugging: Print the result structure
#     print("Revise Code Result Structure:", revise_code_result)

#     revised_code = revise_code_result['result']

#     print("Revised Code:")
#     print(revised_code)

# if __name__ == "__main__":
#     main()





# import os
# from dotenv import load_dotenv
# from crewai import Agent, Task, Crew, Process
# from tools.calculator_tools import CalculatorTools
# from tools.search_tools import SearchTools
# from langchain_openai import AzureChatOpenAI

# load_dotenv()

# class CodeGenerationAgents():
#     def __init__(self):
#         self.llm = AzureChatOpenAI(
#             openai_api_version="2024-05-01-preview",
#             azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#             azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY")
#         )

#     def code_generator(self):
#         return Agent(
#             role='Code Generator',
#             goal='Generate agent code based on user input.',
#             backstory='You are a proficient code generator with extensive knowledge of CrewAI framework.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

#     def code_evaluator(self):
#         return Agent(
#             role='Code Evaluator',
#             goal='Evaluate the generated code and provide feedback.',
#             backstory='You are an experienced code reviewer with an eye for detail.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

#     def code_reviser(self):
#         return Agent(
#             role='Code Reviser',
#             goal='Revise the code based on feedback from the Code Evaluator.',
#             backstory='You are adept at refining code to perfection based on evaluations.',
#             tools=[
#                 SearchTools.search_internet,
#             ],
#             llm=self.llm,
#             verbose=True
#         )

# def main():
#     agents = CodeGenerationAgents()

#     # Runtime user input
#     user_input = input("Please enter the type of agent you want to create: ")

#     # Create the agents
#     code_generator_agent = agents.code_generator()
#     code_evaluator_agent = agents.code_evaluator()
#     code_reviser_agent = agents.code_reviser()

#     # Tasks
#     generate_code_task = Task(
#         description='Generate agent code based on the following user input: {user_input}.',
#         expected_output='The agent code in Python format.',
#         agent=code_generator_agent,
#     )

#     evaluate_code_task = Task(
#         description='Evaluate the following generated code: {generated_code}. Provide detailed feedback on its correctness and quality.',
#         expected_output='Detailed feedback on the generated code.',
#         agent=code_evaluator_agent,
#     )

#     revise_code_task = Task(
#         description='Revise the following code based on the feedback: {feedback}.',
#         expected_output='Revised code in Python format.',
#         agent=code_reviser_agent,
#     )

#     # Forming the crew
#     crew = Crew(
#         agents=[code_generator_agent, code_evaluator_agent, code_reviser_agent],
#         tasks=[generate_code_task],
#         process=Process.sequential
#     )

#     # Kickoff the crew with dynamic user input for code generation
#     generate_code_result = crew.kickoff(inputs={'user_input': user_input})
#     generated_code = generate_code_result['generate_code_task']

#     print("Generated Code:")
#     print(generated_code)

#     # Update the tasks and inputs for evaluation and revision
#     crew.tasks = [evaluate_code_task]
#     evaluate_code_result = crew.kickoff(inputs={'generated_code': generated_code})
#     feedback = evaluate_code_result['evaluate_code_task']

#     print("Feedback:")
#     print(feedback)

#     crew.tasks = [revise_code_task]
#     revise_code_result = crew.kickoff(inputs={'feedback': feedback})
#     revised_code = revise_code_result['revise_code_task']

#     print("Revised Code:")
#     print(revised_code)

# if __name__ == "__main__":
#     main()
