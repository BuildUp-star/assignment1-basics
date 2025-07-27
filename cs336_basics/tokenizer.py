class Tokenizer:
    def __init__(self, vocab:dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens:list[str] | None = None) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(cls, vocab_filepath:str, merges_filepath:str, special_tokens:list[str] | None = None):
        """
        Load a tokenizer from vocab and merges files.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        
        with open(merges_filepath, "r", encoding="utf-8") as f:
            merges = [tuple(line.strip().split()) for line in f.readlines()]

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
    
    def decode(self, ids: list[int]) -> str: