from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model1=ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=0.9
)

model2=ChatGoogleGenerativeAI(
    model='gemini-3.8-flash',
    temperature=0.8
)

parser1=StrOutputParser()

class schema(BaseModel):
    code: str = Field(
        description="The complete corrected C++ code only, without Markdown fences."
    )

parser2=PydanticOutputParser(pydantic_object=schema)

template1=PromptTemplate(
    template='''Check whether the following C++ code is correct.
            If incorrect:
            - Explain the errors.
            - Explain how to fix them.

            If correct:
            - State that the code is correct.

            Code:
            {code}
            ''',
    input_variables=['code']
)

template2=PromptTemplate(
    template='''Return the complete corrected code in the {code} field.
    Requirements:
    - Preserve the original program's intent.
    - Fix compilation errors.
    - Include all required headers.
    - Do not include explanations or Markdown code fences in {code}.
    - Return valid, compilable C++ code. \n {format_instruction}''',
    input_variables=['code'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

runnable=RunnableParallel({
    'check': template1 | model1 | parser1,
    'code': template2 | model2 | parser2
})

code="""
#include<bits/stdc++.h>
using namespace std;
#define int long long
#define pb push_back
const int N=1e9+7;
signed main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    cout.tie(NULL);
    print("Hello World")
}
"""

result= runnable.invoke({'code':code})

print(result)