from importlib.metadata import version 
import tiktoken 
import torch 
from torch.utils.data import Dataset, DataLoader

tokenizer = tiktoken.get_encoding("gpt2")

with open("./files/the-verdict.txt", "r", encoding="utf-8") as f: 
    raw_text = f.read() 

enc_text = tokenizer.encode(raw_text) 
print(len(enc_text))
enc_sample = enc_text[50:]


context_size = 4 #1 
# x = enc_sample[:context_size] 
# y = enc_sample[1:context_size+1] 
# print(f"x: {x}") 
# print(f"y: {y}")

# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(context, "---->", desired)


# for i in range(1, context_size+1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt) #1         
        for i in range(0, len(token_ids) - max_length, stride): #2             
            input_chunk = token_ids[i:i + max_length] 
            target_chunk = token_ids[i + 1: i + max_length + 1] 
            self.input_ids.append(torch.tensor(input_chunk)) 
            self.target_ids.append(torch.tensor(target_chunk)) 
            
    def __len__(self): #3         
        return len(self.input_ids) 
    
    def __getitem__(self, idx): #4         
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2") #1     
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride) #2     
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, #3         
                            num_workers=num_workers #4     
                        )
    return dataloader  

# dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False) 
# data_iter = iter(dataloader) #1 
# first_batch = next(data_iter) 
# print(first_batch)

vocab_size = 50257 
output_dim = 256 
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4 
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False) 
data_iter = iter(dataloader) 
inputs, targets = next(data_iter) 
print("Token IDs:\n", inputs) 
print("\nInputs shape:\n", inputs.shape)