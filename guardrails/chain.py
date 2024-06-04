import os

from langchain_openai import OpenAI

from rich import print
from langchain_community.llms import OpenAI
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser
from langchain_core.prompts import PromptTemplate

# インスタンスの作成
llm = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-3.5-turbo-1106",
    temperature=0.2,
)

# Guardrailの設定
rail_spec = """
<rail version="0.1">

<output>
    <object name="patient_info">
        <string name="gender" description="Patient's gender" />
        <integer name="age" format="valid-range: 0 100" />
        <string name="symptoms" description="Symptoms that the patient is currently experiencing" />
    </object>
</output>

<prompt>

Given the following doctor's notes about a patient, please extract a dictionary that contains the patient's information.

${doctors_notes}

${gr.complete_json_suffix_v2}
</prompt>
</rail>
"""

# 文字列をセットしGuardrailsOutputParserクラスを出力
output_parser = GuardrailsOutputParser.from_rail_string(rail_str=rail_spec, api=llm)

# 入力データを解析して検証し新しいモデルを作成するプロンプトを作成
prompt = PromptTemplate(
    template=output_parser.guard.prompt.escape(),
    input_variables=output_parser.guard.prompt.variable_names,
)

# インスタンスの作成
model = OpenAI(temperature=0)

# インプット
doctors_notes = """
49 y/o Male with chronic macular rash to face & hair, worse in beard, eyebrows & nares.
Itchy, flaky, slightly scaly. Moderate response to OTC steroid cream
"""

# 実行
output = model(prompt.format_prompt(doctors_notes=doctors_notes).to_string())

print(output_parser.parse(output))
