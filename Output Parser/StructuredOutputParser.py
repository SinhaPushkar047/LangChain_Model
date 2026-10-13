from langchain_groq import ChatGroq
from langchain_classic.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model=ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=1.6
)

schema=[
    ResponseSchema(name='Fact1',description='Fact 1 about the topic'),
    ResponseSchema(name='Fact2',description='Fact 2 about the topic'),
    ResponseSchema(name='Fact3',description='Fact 3 about the topic'),
    ResponseSchema(name='Fact4',description='Fact 4 about the topic'),
    ResponseSchema(name='Fact5',description='Fact 5 about the topic'),
]

parser=StructuredOutputParser.from_response_schemas(schema)

template=PromptTemplate(
    template="Give information about the {topic} in 5 points having sufficient information about it \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

chain= template | model | parser

result=chain.invoke({'topic':'Black Hole'})

print(result)
