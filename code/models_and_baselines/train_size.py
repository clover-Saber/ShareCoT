import argparse,json
from pathlib import Path
from protocol import write_json

def ensure_chat_template(tokenizer,canonical):
    if tokenizer.chat_template:return False
    tokenizer.chat_template=canonical;return True

def tokenizer_model_identity(tokenizer):
    return json.loads(tokenizer.backend_tokenizer.to_str())['model']

def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--model-dir', required=True, type=Path)
    parser.add_argument('--canonical-tokenizer-dir', type=Path)
    known, remaining = parser.parse_known_args()
    from transformers import AutoTokenizer
    import train_llm
    original = AutoTokenizer.from_pretrained
    root = known.root
    def load(*args,**kwargs):
        tokenizer=original(*args,**kwargs);used=not bool(tokenizer.chat_template)
        canonical_source = None
        if used:
            if known.canonical_tokenizer_dir is None:
                raise ValueError('--canonical-tokenizer-dir is required when the model has no chat template')
            model_dir = known.model_dir
            canonical_source = known.canonical_tokenizer_dir / 'tokenizer_config.json'
            canonical_tokenizer = original(known.canonical_tokenizer_dir, local_files_only=True)
            if not canonical_tokenizer.chat_template:
                raise ValueError('Canonical tokenizer has no chat template')
            if tokenizer_model_identity(tokenizer)!=tokenizer_model_identity(canonical_tokenizer):raise ValueError('Sized checkpoint embedded BPE vocabulary/merges differ from canonical tokenizer')
            tokenizer=canonical_tokenizer
            vocab_size=json.loads((model_dir/'config.json').read_text(encoding='utf-8'))['vocab_size']
            if max(tokenizer.get_vocab().values())>=vocab_size:raise ValueError('Canonical tokenizer has IDs outside sized model embeddings')
        identity={'model_dir':str(Path(known.model_dir).resolve()),'canonical_tokenizer_used':used,'canonical_source':str(canonical_source) if used else None,'eos_token_id':tokenizer.eos_token_id,'pad_token_id':tokenizer.pad_token_id,'reason':'local sized checkpoint lacks tokenizer_config; use complete matched Qwen2.5-7B family tokenizer only after embedded BPE vocabulary/merge equality and embedding ID checks'}
        path=root/'tokenizer_template_provenance.json'
        if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=identity:raise ValueError('Tokenizer template provenance changed')
        write_json(path,identity);return tokenizer
    AutoTokenizer.from_pretrained=staticmethod(load)
    try:return train_llm.main(['--root', str(root), '--model-dir', str(known.model_dir), *remaining])
    finally:AutoTokenizer.from_pretrained=original

if __name__=='__main__':raise SystemExit(main())
