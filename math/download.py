from transformers import AutoModel, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
save_path = "./models/Qwen2.5-7B"

# 下载模型和 tokenizer
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)