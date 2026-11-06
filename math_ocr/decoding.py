"""Optional validation-selected beam search with bounded syntax reranking."""
import torch


@torch.inference_mode()
def generate(model,package,images,widths):
    maximum=package.get('max_decode',48);beam=package.get('beam_width',1)
    if beam==1:return model.generate(images,widths,maximum)
    paths,scores=model.generate_candidates(images,widths,maximum,beam,package.get('length_alpha',.6))
    paths=paths.cpu();order=scores.argsort(dim=1,descending=True).cpu();chosen=[]
    if package.get('syntax')=='physics-v1':from physics_syntax import parse
    else:from slash_semantics import parse
    for sample,rank in zip(paths,order):
        selected=sample[rank[0]]
        for index in rank:
            ids=sample[index].tolist()
            if 2 not in ids:continue
            words=[package['vocab'][i] for i in ids[:ids.index(2)]]
            try:parse(words);selected=sample[index];break
            except (ValueError,RecursionError):continue
        chosen.append(selected)
    return torch.stack(chosen)
