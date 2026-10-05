import torch
from torch import Tensor
import heapq
from itertools import pairwise

class BPETokenizer:
    def __init__(self, chars, merge_table=None):
        self.chars = chars.copy()

        self.idx = len(self.chars)

        # lower idx pair may get merged into higher idx, but not vice versa
        # Fresh dict per instance; copy caller's table so we don't mutate it.
        self.merge_table = dict(merge_table) if merge_table is not None else {}
        self.reverse_table = {}
        for pair in self.merge_table:
            value = self.merge_table[pair]
            self.reverse_table[value] = pair

    def _register_pair(self, pair):
        new_idx = self.idx
        self.idx += 1

        assert pair not in self.merge_table
        self.merge_table[pair] = new_idx
        self.reverse_table[new_idx] = pair
        return new_idx


    """
    1. encode into [1, 7, 3, 1, 7, 3, 1, 7]
    2. try to find common pattern like [1, 7]
    3. replace into [42, 3, 42, 3, 42]
    4. repeat until nothing left to merge or we reach vocab_size
    """
    def train(self, documents, vocab_size):
        # encode char -> idx
        encoded_list = [] # list of list of index
        for document in documents:
            encoded_list.append([self.chars.index(c) for c in document])

        while len(self.merge_table) + len(self.chars) < vocab_size:
            # loop the string to find most common pair
            counter = {}
            common = None

            for encoded in encoded_list:
                for pair in pairwise(encoded):
                    if pair not in counter:
                        counter[pair] = 0
                    counter[pair] += 1
                    if common == None or counter[common] < counter[pair]:
                        common = pair

            # no adjacent pairs remain
            if common == None:
                break

            # found the common pair, register it
            merge_idx = self._register_pair(common)
            
            # then replace original encoded string
            merged_list = []
            for encoded in encoded_list:
                merged = []
                i = 0
                while i < len(encoded):
                    # found a merge
                    if i+1 < len(encoded) and encoded[i] == common[0] and encoded[i+1] == common[1]:
                        merged.append(merge_idx)
                        i += 2
                    else:
                        merged.append(encoded[i])
                        i += 1
                merged_list.append(merged)
            encoded_list = merged_list

    class Token:
        def __init__(self, value, pos, prev, next, merged = False):
            self.value = value  # encoded value
            self.pos = pos      # for breaking tie
            self.prev = prev
            self.next = next
            self.merged = merged 

        def __repr__(self):
          return f"Token(pos: {self.pos}, value: {self.value})"

    # merge a token with its right sibling -> return merged token
    def merge(self, token: Token):
        new_value = self.lookup_merge(token)
        assert new_value != None

        # neighbors of the merged tokens
        left = token.prev
        sibling = token.next
        right = token.next.next

        # create a new merged token and update its neighbors
        merged = BPETokenizer.Token(new_value, token.pos, left, right)
        if left != None:
            left.next = merged
        if right != None:
            right.prev = merged

        # mark the 2 merged tokens as merged
        token.merged = True
        sibling.merged = True

        return merged

    # return rank if mergeable, else None
    def lookup_merge(self, token: Token):
        if token == None or token.next == None:
            return None
        if token.merged or token.next.merged:
            return None

        pair = (token.value, token.next.value)
        if pair not in self.merge_table:
            return None
        return self.merge_table[pair]

    def attempt_merge(self, pq, token):
        rank = self.lookup_merge(token)
        if rank != None:
            heapq.heappush(pq, (rank, token.pos, token, token.next))

    # string -> value
    def encode(self, document) -> Tensor:
        if len(document) == 0:
            return torch.tensor([])
        
        # first create tokens
        tokens = []
        for pos in range(len(document)):
            c = document[pos]
            tokens.append(BPETokenizer.Token(self.chars.index(c), pos, prev = None, next = None))
        for l, r in pairwise(tokens):
            l.next = r
            r.prev = l

        # refreshs and sort by merge priority
        pq = [] # (rank, token)
        for token in tokens:
            self.attempt_merge(pq, token)
        head = tokens[0]
        
        # now keep merging everything thats mergeable, similar to a DFS
        while pq:
            (_, _, token, expected_next) = heapq.heappop(pq)

            # either token got merged, or token.next got merged
            if token.merged or expected_next.merged:
                continue
            
            # merge the token with its sibling
            merged = self.merge(token)

            # the current merge may trigger left neighbor and merged token to become mergeable
            left = merged.prev
            self.attempt_merge(pq, left)
            self.attempt_merge(pq, merged)

            # merged token may be the head
            if merged.prev == None:
                head = merged

        # finally collect all Tokens from head
        cur = head
        ret = []
        while cur != None:
            ret.append(cur.value)
            cur = cur.next

        return torch.tensor(ret)

    # idea is to keep expanding merged token until it is down to a single character
    def decode_single(self, value) -> str:
        # single char - termination
        if value not in self.reverse_table:
            return self.chars[value]

        # merged
        left, right = self.reverse_table[value]
        return self.decode_single(left) + self.decode_single(right)

    # tensor -> string
    def decode(self, encoded: Tensor) -> str:
        s = ""
        for value in encoded:
            s += self.decode_single(value.item())
        return s

class CharacterTokenizer:
    SEP = "\n"

    def __init__(self, chars):
        self.chars = chars.copy()
        self.chars.append(self.SEP)
        self.ctoi = lambda c: self.chars.index(c)
        self.itoc = lambda i: self.chars[i]
        self.vocab_size = len(self.chars)
        self.sep_id = self.ctoi(self.SEP)

    def encode(self, name: str) -> Tensor:
        return torch.tensor([self.ctoi(c) for c in list(name)], dtype = torch.long)

    def decode(self, encoded: Tensor) -> str:
        assert encoded.ndim == 1
        return "".join([self.itoc(i) for i in encoded])
