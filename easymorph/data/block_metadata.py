from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BlockMetadata:
    name: str
    type: str
    # Start of block content: the stream position after the block's name,
    # type and length are read from the stream.
    block_start_position: Optional[int]
    # Length of block content, i.e. the length of the block without its name,
    # type and length records.
    block_length: Optional[int]
