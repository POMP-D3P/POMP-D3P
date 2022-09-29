import random
import copy
import numpy as np
from operator import itemgetter


class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, mask, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = tuple(
            [copy.deepcopy(x) for x in [state, action, reward, next_state, mask, done]]
        )
        # print(self.buffer[self.position])
        self.position = (self.position + 1) % self.capacity
        # print(self.position)

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[
                : len(self.buffer) - self.position
            ]
            self.buffer[: len(batch) - len(self.buffer) + self.position] = batch[
                len(self.buffer) - self.position :
            ]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask, done

    def sample_near(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        # batch = random.sample(self.buffer, batch_size)
        batch = self.buffer[-batch_size:]
        state, action, reward, next_state, mask, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, mask, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)


class ReplayMemory_without_mask:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = tuple(
            [copy.deepcopy(x) for x in [state, action, reward, next_state, done]]
        )
        # print(self.buffer[self.position])
        self.position = (self.position + 1) % self.capacity
        # print(self.position)

    def push_batch(self, batch):
        if len(self.buffer) < self.capacity:
            append_len = min(self.capacity - len(self.buffer), len(batch))
            self.buffer.extend([None] * append_len)

        if self.position + len(batch) < self.capacity:
            self.buffer[self.position : self.position + len(batch)] = batch
            self.position += len(batch)
        else:
            self.buffer[self.position : len(self.buffer)] = batch[
                : len(self.buffer) - self.position
            ]
            self.buffer[: len(batch) - len(self.buffer) + self.position] = batch[
                len(self.buffer) - self.position :
            ]
            self.position = len(batch) - len(self.buffer) + self.position

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def sample_all_batch(self, batch_size):
        idxes = np.random.randint(0, len(self.buffer), batch_size)
        batch = list(itemgetter(*idxes)(self.buffer))
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def return_all(self):
        return self.buffer

    def __len__(self):
        return len(self.buffer)
