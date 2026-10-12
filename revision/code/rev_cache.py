# Build the embedding caches used by every run (data/emb_*.pt).
import sys
from rev_lib import embedding_cache, get_device, effective_rank
dev = get_device()
for ds, m in [("ag_news", "bert-base-uncased"), ("sst2", "bert-base-uncased"),
              ("ag_news", "distilbert-base-uncased")]:
    C = embedding_cache(ds, m, dev)
    print(ds, m, tuple(C["train"].shape), tuple(C["test"].shape),
          "classes", C["n_classes"], "effrank(test)", effective_rank(C["test"]),
          flush=True)
