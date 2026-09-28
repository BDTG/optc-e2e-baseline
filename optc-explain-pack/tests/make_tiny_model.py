import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

CORPUS = [
    'You are a SOC analyst. Explain the alert. {"verdict": "malicious", "evidence": ["E1.TargetImage=C:\\\\Windows\\\\system32\\\\lsass.exe"], "technique": "T1003 OS Credential Dumping"}',
    '[E0] Event 1 Process Create | Image: C:\\Windows\\System32\\cmd.exe | CommandLine: cmd /c whoami | ParentImage: C:\\Windows\\explorer.exe',
    '[E1] Event 10 Process accessed | SourceImage: C:\\Tools\\procdump.exe | TargetImage: C:\\Windows\\system32\\lsass.exe | GrantedAccess: 0x1010',
    '{"verdict": "benign", "evidence": [], "technique": "none"} T1059 T1055 T1543 T1490 T1553 T1574 T1047 T1218 T1036 N/A',
] * 50


def build(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tk = Tokenizer(models.BPE(unk_token=None))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    special = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tr = trainers.BpeTrainer(vocab_size=600, special_tokens=special,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(CORPUS, tr)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  bos_token=None, unk_token=None)
    tok.chat_template = TEMPLATE
    tok.save_pretrained(out)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    torch.manual_seed(0)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return out


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-qwen"))
