"""
Run simply with
$ pytest
"""
import argparse
import os
import pytest # pip install pytest
import subprocess

import torch
from model import ModelArgs, Transformer
from tokenizer import Tokenizer
from config import TOKENIZER_BIN

def test_argmax_inference(ckpt_dir):
    """
    Only the simplest test for now: run inference with temperature 0 
    (for determinism) in both C and PyTorch, and see that the sampled tokens 
    are the same.
    """
    # test_ckpt_dir = "out/demo0-en-llama2" # TODO create a dummy test checkpoint for this?

    # run C version
    model_path = os.path.join(ckpt_dir, "model.bin")
    
    command = ["./run", model_path, "-k", TOKENIZER_BIN, "-t", "0.0"]
    proc = subprocess.Popen(command, stdout=subprocess.PIPE)
    c_lines = []
    for line in proc.stdout:
        line = line.decode('utf-8').strip()
        # if token == "<s>" or token =="</s>": continue
        # token = int(token)
        c_lines.append(line)
    proc.wait()
    #print(c_tokens)

    # run PyTorch version
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = ModelArgs(**checkpoint['model_args'])
    model = Transformer(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)
    x = torch.tensor([[1]], dtype=torch.long, device=device) # 1 is BOS
    with torch.inference_mode():
        y = model.generate(x, max_new_tokens=gptconf.max_seq_len, temperature=0.0)
    # 截断重复
    pt_tokens = y[0]
    bosIdx = torch.where(pt_tokens==1)[0]
    if len(bosIdx)>1:
        pt_tokens = pt_tokens[:bosIdx[1]+1]
    pt_tokens = pt_tokens.tolist()
    pt_lines = tokenizer.decode(pt_tokens).split("\n")
    for i in range(len(pt_lines)):
        pt_line = pt_lines[i].strip()
        c_line = c_lines[i]
        print(f"c>>{c_line}\np>>{pt_line}\n")
        assert pt_line == c_line, "Not Same."
    print("it's good.")


if __name__ == "__main__":
    """Usage:
        python test_all.py out/
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_dir", type=str, default=None, help="待验证的训练模型目录, eg: out/")
    args = parser.parse_args()
    print(f"TOKENIZER_BIN={TOKENIZER_BIN}")
    print(f"ckpt_dir={args.ckpt_dir}")
    tokenizer  = Tokenizer()
    test_argmax_inference(args.ckpt_dir)
    