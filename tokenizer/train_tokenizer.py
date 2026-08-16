import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
SPECIAL=["<|bos|>","<|eos|>","<|pad|>","<|system|>","<|user|>","<|assistant|>"]
p=argparse.ArgumentParser(); p.add_argument("--input",nargs="+",required=True); p.add_argument("--output",default="tokenizer/tokenizer.json"); a=p.parse_args()
t=Tokenizer(models.BPE(unk_token=None)); t.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False); t.decoder=decoders.ByteLevel()
t.train(a.input,trainers.BpeTrainer(vocab_size=16384,min_frequency=2,special_tokens=SPECIAL,initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
Path(a.output).parent.mkdir(parents=True,exist_ok=True); t.save(a.output)
for sample in ["hello", "what does ram actaully do", "<button>Hi</button>", "const x = () => 1;", "for i in range(3):\n    print(i)", "héllo 🐍"]: assert t.decode(t.encode(sample).ids)==sample
print(f"saved={a.output} vocab={t.get_vocab_size()}")

