"""A small, explicitly attention-free CNN/BiLSTM/LSTM image-to-LaTeX model."""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


@dataclass
class ModelConfig:
    vocab_size: int
    height: int = 64
    feature_size: int = 192
    encoder_hidden: int = 192
    decoder_hidden: int = 256
    embedding_size: int = 96
    layers: int = 2
    dropout: float = 0.1
    context_bins: int = 0
    auxiliary: bool = False
    local_features: int = 0
    raw_context: bool = False
    refinement_blocks: int = 0


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ResidualRefinement(nn.Module):
    def __init__(self,channels=128):
        super().__init__()
        self.layers=nn.Sequential(conv(channels,channels),nn.Conv2d(channels,channels,3,padding=1,bias=False),nn.BatchNorm2d(channels))
        nn.init.zeros_(self.layers[-1].weight)
    def forward(self,x):return torch.relu(x+self.layers(x))


class FormulaOCR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        c = config
        self.cnn = nn.Sequential(
            conv(1, 24), nn.MaxPool2d(2),
            conv(24, 48), nn.MaxPool2d(2),
            conv(48, 96), conv(96, 128), nn.MaxPool2d((2, 1)),
        )
        if c.refinement_blocks:self.refinement=nn.Sequential(*(ResidualRefinement() for _ in range(c.refinement_blocks)))
        # Flatten vertical features, preserving top/bottom position within every column.
        self.projection = nn.Sequential(nn.Linear(128 * (c.height // 8), c.feature_size),
                                        nn.LayerNorm(c.feature_size), nn.ReLU())
        self.encoder = nn.LSTM(c.feature_size, c.encoder_hidden, c.layers,
                               batch_first=True, bidirectional=True, dropout=c.dropout)
        context = c.encoder_hidden * 2
        if c.context_bins:
            self.context_projection=nn.Sequential(nn.Linear(context*c.context_bins,context),nn.LayerNorm(context),nn.Tanh())
            context*=2
        if c.raw_context:
            if not c.context_bins:raise ValueError('Raw context requires fixed context bins')
            context+=c.encoder_hidden*2*c.context_bins
        self.init_h = nn.Linear(context, c.layers * c.decoder_hidden)
        self.init_c = nn.Linear(context, c.layers * c.decoder_hidden)
        self.embedding = nn.Embedding(c.vocab_size, c.embedding_size, padding_idx=0)
        if c.local_features:
            if not c.auxiliary:raise ValueError('Fixed local features require the output length head')
            self.local_projection=nn.Sequential(nn.Linear(c.encoder_hidden*2*3,c.local_features),nn.Tanh())
        self.decoder = nn.LSTM(c.embedding_size + context+c.local_features, c.decoder_hidden, c.layers,
                               batch_first=True, dropout=c.dropout)
        self.output = nn.Linear(c.decoder_hidden, c.vocab_size)
        if c.auxiliary:
            self.symbol_presence=nn.Linear(context,c.vocab_size)
            self.output_length=nn.Linear(context,73)

    def encode(self, images, widths,return_sequence=False):
        features=self.cnn(images)
        if self.config.refinement_blocks:features=self.refinement(features)
        features=features.permute(0,3,1,2).flatten(2)
        features = self.projection(features)
        lengths = (widths // 4).clamp(min=1).cpu()
        packed = pack_padded_sequence(features, lengths, batch_first=True, enforce_sorted=False)
        encoded, (hidden, _) = self.encoder(packed)
        z = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        if self.config.context_bins or return_sequence:
            sequence,_=pad_packed_sequence(encoded,batch_first=True)
        if self.config.context_bins:
            # Fixed equally spaced positions, independent of decoder state and
            # image content: no learned attention weights or token alignment.
            positions=torch.linspace(0,1,self.config.context_bins,device=images.device)[None,:]*(lengths.to(images.device)[:,None]-1)
            lo=positions.long();hi=(lo+1).clamp(max=sequence.size(1)-1)
            indexes=torch.arange(images.size(0),device=images.device)[:,None]
            samples=sequence[indexes,lo]+(sequence[indexes,hi]-sequence[indexes,lo])*(positions-lo).unsqueeze(-1)
            z=torch.cat([z,self.context_projection(samples.flatten(1))],dim=-1)
            if self.config.raw_context:z=torch.cat([z,samples.flatten(1)],dim=-1)
        batch = images.size(0)
        shape = (batch, self.config.layers, self.config.decoder_hidden)
        h = self.init_h(z).tanh().reshape(shape).transpose(0, 1).contiguous()
        cell = self.init_c(z).tanh().reshape(shape).transpose(0, 1).contiguous()
        if return_sequence:return z,(h,cell),sequence,lengths.to(images.device)
        return z, (h, cell)

    def fixed_local(self,sequence,source_lengths,word_lengths,steps):
        """Three fixed sample locations, with no query/key scores or learned weights.

        Training uses the supervised total length; inference uses the auxiliary
        length classifier. Predicted length is the only image-dependent scalar
        in the position schedule. It is not ground-truth length at inference.
        """
        steps=steps.to(sequence.device).float()[None,:,None]
        span=(source_lengths.float()-1)[:,None,None]
        count=word_lengths.float().clamp(min=1)[:,None,None]
        offsets=torch.tensor([-1.,0.,1.],device=sequence.device)[None,None,:]
        positions=steps/(count-1).clamp(min=1)*span+offsets*span/count
        positions=positions.clamp(min=0).minimum(span)
        lo=positions.long();hi=(lo+1).minimum((source_lengths-1)[:,None,None])
        batch=torch.arange(sequence.size(0),device=sequence.device)[:,None,None]
        sampled=sequence[batch,lo]+(sequence[batch,hi]-sequence[batch,lo])*(positions-lo).unsqueeze(-1)
        return self.local_projection(sampled.flatten(2))

    def teacher_decode(self,z,state,previous_tokens,local=None):
        context = z.unsqueeze(1).expand(-1, previous_tokens.size(1), -1)
        inputs = torch.cat([self.embedding(previous_tokens), context], dim=-1)
        if local is not None:inputs=torch.cat([inputs,local],dim=-1)
        decoded, _ = self.decoder(inputs, state)
        return self.output(decoded)

    def teacher_forward(self,images,widths,previous_tokens):
        if self.config.local_features:
            z,state,sequence,lengths=self.encode(images,widths,True)
            words=(previous_tokens.ne(0).sum(1)-1-previous_tokens.eq(2).sum(1)).clamp(min=1)
            if self.training:
                predicted=self.output_length(z).argmax(-1).clamp(min=1)
                words=torch.where(torch.rand(len(words),device=images.device)<.5,predicted,words)
            local=self.fixed_local(sequence,lengths,words,torch.arange(previous_tokens.size(1),device=images.device))
        else:z,state=self.encode(images,widths);local=None
        return self.teacher_decode(z,state,previous_tokens,local),z

    def forward(self, images, widths, previous_tokens):
        return self.teacher_forward(images,widths,previous_tokens)[0]

    def forward_with_aux(self,images,widths,previous_tokens):
        decoded,z=self.teacher_forward(images,widths,previous_tokens)
        return decoded,self.symbol_presence(z),self.output_length(z)

    @torch.inference_mode()
    def generate(self, images, widths, max_tokens=72):
        if self.config.local_features:
            z,state,sequence,lengths=self.encode(images,widths,True)
            word_lengths=self.output_length(z).argmax(-1).clamp(min=1)
        else:z,state=self.encode(images,widths)
        token = torch.full((images.size(0), 1), 1, device=images.device, dtype=torch.long)
        finished = torch.zeros(images.size(0), dtype=torch.bool, device=images.device)
        outputs = []
        for step in range(max_tokens):
            inputs = torch.cat([self.embedding(token), z.unsqueeze(1)], dim=-1)
            if self.config.local_features:
                inputs=torch.cat([inputs,self.fixed_local(sequence,lengths,word_lengths,torch.tensor([step],device=images.device))],dim=-1)
            decoded, state = self.decoder(inputs, state)
            logits = self.output(decoded[:, -1])
            logits[:, 0:2] = -torch.inf  # PAD/BOS are never output symbols.
            token = logits.argmax(-1)
            token = torch.where(finished, torch.full_like(token, 2), token)
            outputs.append(token)
            finished |= token.eq(2)
            token = token.unsqueeze(1)
            if finished.all():
                break
        return torch.stack(outputs, dim=1)

    def config_dict(self):
        return asdict(self.config)

    @torch.inference_mode()
    def generate_candidates(self,images,widths,max_tokens=72,beam_width=3,length_alpha=.6):
        """Beam search over the same fixed context; no reference tokens."""
        batch=images.size(0);k=beam_width
        if self.config.local_features:
            z,state,sequence,source_lengths=self.encode(images,widths,True)
            word_lengths=self.output_length(z).argmax(-1).clamp(min=1).repeat_interleave(k,0)
            sequence=sequence.repeat_interleave(k,0);source_lengths=source_lengths.repeat_interleave(k,0)
        else:z,state=self.encode(images,widths)
        z=z.repeat_interleave(k,0);state=tuple(s.repeat_interleave(k,1) for s in state)
        scores=torch.full((batch,k),-torch.inf,device=images.device);scores[:,0]=0
        paths=torch.empty((batch,k,0),device=images.device,dtype=torch.long)
        token=torch.ones((batch*k,1),device=images.device,dtype=torch.long)
        done=torch.zeros((batch,k),device=images.device,dtype=torch.bool)
        lengths=torch.zeros((batch,k),device=images.device,dtype=torch.long)
        for step in range(max_tokens):
            inputs=torch.cat([self.embedding(token),z.unsqueeze(1)],dim=-1)
            if self.config.local_features:
                inputs=torch.cat([inputs,self.fixed_local(sequence,source_lengths,word_lengths,torch.tensor([step],device=images.device))],dim=-1)
            decoded,next_state=self.decoder(inputs,state)
            logits=self.output(decoded[:,-1]);logits[:,:2]=-torch.inf
            logp=logits.log_softmax(-1).view(batch,k,-1)
            logp=logp.masked_fill(done[:,:,None],-torch.inf)
            logp[:,:,2]=torch.where(done,torch.zeros_like(scores),logp[:,:,2])
            combined=scores[:,:,None]+logp
            scores,indexes=combined.flatten(1).topk(k,dim=1)
            parents=indexes//self.config.vocab_size;words=indexes%self.config.vocab_size
            offsets=(torch.arange(batch,device=images.device)[:,None]*k+parents).flatten()
            state=tuple(s.index_select(1,offsets) for s in next_state)
            paths=torch.cat([paths.gather(1,parents[:,:,None].expand(-1,-1,paths.size(-1))),words[:,:,None]],dim=-1)
            previous_done=done.gather(1,parents)
            lengths=lengths.gather(1,parents)+(~previous_done).long()
            done=previous_done|words.eq(2);token=words.reshape(-1,1)
            if done.all():break
        normalized=scores/lengths.clamp(min=1).float().pow(length_alpha)
        return paths,normalized
