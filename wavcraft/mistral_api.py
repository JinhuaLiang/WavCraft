import torch
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer


class _chatLLM:
    def __init__(self, model_id):
        self.messages = []
        self.model, self.tokenizer = self.build_model(model_id)

    def create_new_session(self):
        self.messages = []


class ChatMistral(_chatLLM):
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        super().__init__(model_id)
        self.device = self.model.device


    def get_response(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})

        encodes = self.tokenizer.apply_chat_template(self.messages, return_tensors="pt").to(self.device)

        generated_ids = self.model.generate(encodes, max_new_tokens=1000, do_sample=True)
        decoded = self.tokenizer.batch_decode(generated_ids)[-1]

        response = self.extract_response(decoded)
        self.messages.append({"role": "assistant", "content": response})  # Update conversation

        return response
    

    def build_model(self, model_id="mistralai/Mistral-7B-Instruct-v0.2"):
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            # 8-bit quantization
            load_in_8bit=True,
            # 4-bit quantization
            # load_in_4bit=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.float16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16, 
            # attn_implementation="flash_attention_2",  # NOTE: cannot use with V100
            quantization_config=quantization_config,
            device_map="auto")
        
        # [Updated 23-04-2024] mistral built-in tokenizer has the same function as huggingface tokenizer
        # Ones can stick to huggingface tokenizer UNLESS tool calling is required.
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        return model, tokenizer


    def extract_response(self, responses):
        # Split the interaction by "</s>" to separate each round, and take the last non-empty round if any
        rounds = [r for r in responses.split("</s>") if r.strip()]
        # Split response by "[/INST]"
        last_reponse = rounds[-1].strip().split("[/INST]")[-1].strip()

        return last_reponse

if __name__ == "__main__":
    llm = ChatMistral()
    llm.messages = [
        {"role": "user", "content": "What is your favourite condiment?"},
        {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
    ]
    re = llm.get_response("Do you have mayonnaise recipes?")
    print(re)
    import ipdb; ipdb.set_trace()
