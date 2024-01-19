import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

base_model_id = "mistralai/Mixtral-8x7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, add_bos_token=True, trust_remote_code=True)

from peft import PeftModel

ft_model = PeftModel.from_pretrained(base_model, "pawan2411/mixtral-journal-finetune")

import json
q_file = open("Test_Numerical_Reasoning.json", encoding='utf-8')
cs_file = open("sres.txt", "w", encoding='utf-8')
data = json.load(q_file)
done = 0
for dp in data[2000: ]:
  human_t = "context: "+dp["news"]+"\n"+"question: "+dp["masked headline"]
  model_input = tokenizer(human_t, return_tensors="pt").to("cuda:0")
  ft_model.eval()
  with torch.no_grad():
    res1 =(tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=512)[0], skip_special_tokens=True,do_sample=True))
  cs_file.write(res1.replace("\n","####")+"\n")
  done =done+1
  print(done,"done")

