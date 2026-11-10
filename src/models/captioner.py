"""RSCaMa construction shared by training, pair inference and caption evaluation."""
from pathlib import Path
import json
import os
import sys


def build_components(source_dir, vocabulary, device):
    import torch
    from src.models.rscama_compat import install_patches
    source = Path(source_dir).resolve()
    if not (source/'model/model_encoder_attMamba.py').is_file():
        raise FileNotFoundError('Run scripts/setup_model_sources.py to acquire the pinned RSCaMa source')
    install_patches(device)
    sys.path.insert(0, str(source))
    previous = Path.cwd()
    try:
        os.chdir(source)
        import clip
        from model.mamba_block import MambaMixer
        from model.model_encoder_attMamba import Encoder, AttentiveEncoder
        from model.model_decoder import DecoderTransformer
        MambaMixer.forward = lambda self,h,h2,cache_params=None,cache_params_2=None: self.cuda_kernels_forward(h,h2)
        _, processor = clip.load('ViT-B/32', device='cpu', jit=False)
        encoder = Encoder('CLIP-ViT-B/32')
        fusion = AttentiveEncoder(n_layers=3, feature_size=[7,7,768], heads=8, dropout=.1)
        decoder = DecoderTransformer(decoder_type='transformer_decoder', embed_dim=768,
            vocab_size=len(vocabulary), max_lengths=42, word_vocab=vocabulary, n_head=8, n_layers=1, dropout=.1)
    finally:
        os.chdir(previous)
    return (encoder, fusion, decoder), processor


class RSCaMaCaptioner:
    def __init__(self, spec, device='cuda'):
        import torch
        self.device, self.torch = device, torch
        checkpoint = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        self.vocab = checkpoint.get('vocab')
        if self.vocab is None:
            if not spec.get('vocab'):
                raise ValueError('Released checkpoints require their matching vocabulary JSON')
            self.vocab = json.loads(Path(spec['vocab']).read_text(encoding='utf-8'))
        if len(self.vocab) != checkpoint['decoder_dict']['wdc.weight'].shape[0]:
            raise ValueError('Caption checkpoint and vocabulary sizes differ')
        self.models, self.processor = build_components(spec['source_dir'], self.vocab, device)
        for model, key in zip(self.models, ['encoder_dict','encoder_trans_dict','decoder_dict']):
            model.load_state_dict(checkpoint[key], strict=True); model.to(device).eval()
        self.inverse = {int(value): word for word, value in self.vocab.items()}

    def describe(self, images):
        from PIL import Image
        arrays = []
        for path in images:
            with Image.open(path) as image:
                arrays.append(self.processor(image.convert('RGB')).unsqueeze(0).to(self.device))
        encoder, fusion, decoder = self.models
        with self.torch.inference_mode():
            first, second = encoder(*arrays)
            tokens = decoder.sample(fusion(first, second), k=1)
        if hasattr(tokens, 'tolist'): tokens = tokens.tolist()
        while tokens and isinstance(tokens[0], list): tokens = tokens[0]
        return ' '.join(self.inverse[int(token)] for token in tokens
                        if self.inverse[int(token)] not in {'<START>','<END>','<NULL>'})
