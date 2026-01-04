import torch.nn.functional as F
from chromadb import Documents, Embeddings
import torch
from jedi.settings import cache_directory
from torch import Tensor
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

cacheDir = './hf_cache'

class Qwen3Embedder:
    def last_token_pool(self, last_hidden_states: Tensor,
        attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-Embedding-0.6B', padding_side='left', cache_dir=cacheDir)
        if torch.cuda.is_available():
            self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', cache_dir=cacheDir, dtype=torch.float16).cuda()
        else:
            self.model = AutoModel.from_pretrained('Qwen/Qwen3-Embedding-0.6B', cache_dir=cacheDir, dtype=torch.float16)

    def __call__(self, input: Documents) -> Embeddings:
        self.batch_dict = self.tokenizer(
            input,
            padding=True,
            truncation=True,
            max_length=100,
            return_tensors="pt",
        )
        self.batch_dict.to(self.model.device)
        self.outputs = self.model(**self.batch_dict)
        self.embeddings = self.last_token_pool(self.outputs.last_hidden_state, self.batch_dict['attention_mask'])

        # normalize embeddings
        self.embeddings = F.normalize(self.embeddings, p=2, dim=1)
        return self.embeddings.tolist()
    def name(self):
        return 'Qwen3Embedder'

class Qwen3Chat:
    model_name = "Qwen/Qwen3-0.6B"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=cacheDir)
        if torch.cuda.is_available():
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto",
                cache_dir=cacheDir
            ).cuda()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype="auto",
                device_map="auto",
                cache_dir=cacheDir
            )

    def __call__(self, question, context):
        # prepare the model input
        prompt = f'''DOCUMENT:
                    {context}
                    
                    QUESTION:
                    {question}
                    
                    INSTRUCTIONS:
                    Answer the users QUESTION using the DOCUMENT text above.
                    Keep your answer ground in the facts of the DOCUMENT.
                    Keep your answer within three sentences.
                    If the DOCUMENT doesn’t contain the facts to answer the QUESTION, say you don't know'''

        messages = [
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=32768
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        # print("thinking content:", thinking_content)
        return content