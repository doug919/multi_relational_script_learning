
import torch


class NegativeSampler:
    def __init__(self, maxlens, n_rels, sample_rel=False):
        self.arg0_max_len, self.arg1_max_len, self.arg2_max_len = maxlens
        self.sample_rel = sample_rel
        self.n_rels = n_rels

    def sampling(self, x):
        # x has to be on CPU, since PyTorch 0.4.0 doesn't support randperm on cuda.
       
        # truncate events
        n_samples = x.shape[0]
        x_neg = x.clone()

        idxs = torch.randperm(n_samples)
        chunks = torch.chunk(idxs, 3) if self.sample_rel else torch.chunk(idxs, 2)
        
        # corrupt e1
        neg_e = x[:, 1:1+1+self.arg0_max_len+self.arg1_max_len]
        neg_e = neg_e[torch.randperm(x.shape[0])]
        x_neg[chunks[0], 1:1+1+self.arg0_max_len+self.arg1_max_len] = \
                neg_e[chunks[0]]

        # corrupt e2
        neg_e = x[:, 1+1+self.arg0_max_len+self.arg1_max_len:]
        neg_e = neg_e[torch.randperm(x.shape[0])]
        x_neg[chunks[1], 1+1+self.arg0_max_len+self.arg1_max_len:] = \
                neg_e[chunks[1]]

        if self.sample_rel:
            # corrupt relations
            neg_rels = torch.randint_like(chunks[2], 0, self.n_rels)
            while True:
                idxs = (neg_rels == x[chunks[2], 0]).nonzero().view(-1)
                if len(idxs) == 0:
                    break
                for i in idxs:
                    neg_rels[i] = torch.randint(0, self.n_rels, (1,))
            x_neg[chunks[2], 0] = neg_rels
        return x_neg
