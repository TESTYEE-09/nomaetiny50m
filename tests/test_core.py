import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from generator.generate import stable_id,prompt_for,cost_usd
from tokenizer.chat import format_chat
from training.model import TinyLM

def test_parameter_target(): assert 45_000_000 <= TinyLM().parameter_count() <= 55_000_000
def test_ids_are_stable(): assert stable_id(1,2)==stable_id(1,2) and stable_id(1,2)!=stable_id(1,3)
def test_chat_template(): assert format_chat([{"role":"user","content":"hi"}],True)=="<|bos|><|user|>hi<|eos|><|assistant|>"
def test_cost(): assert abs(cost_usd({"prompt_tokens":1_000_000,"completion_tokens":1_000_000})-.42)<1e-12
