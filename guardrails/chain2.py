import os

from rich import print
from langchain_openai import OpenAI
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
    <object name="speaker_info">
        <string name="name" description="Speaker's name" />
        <integer name="height" description="Speaker's heigth" />
        <integer name="weight" format="valid-range: 0 1000" />
        <string name="symptoms" description="Symptoms that the speaker is currently experiencing" />
    </object>
</output>

<prompt>

Extract a dictionary containing the speaker's information from the following inputs.

${speaker_input}

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
speaker_input = """
僕の名前は山田太郎だよ、性別は男、僕の体は身長180センチ、体重120キロなんだけど、太り過ぎが悪いと思うよ。
そして僕は糖尿病なんだけどで太り過ぎが原因だと思うよ。
"""

# 実行
output = model(prompt.format_prompt(speaker_input=speaker_input).to_string())

print(output_parser.parse(output))
