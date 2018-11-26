import numpy as np

def load_w2v(path):
    """Loads embedings, returns weight matrix and dict from words to indices."""
    word_vectors, word_idx = [], {}
    with open(path, 'r', encoding="utf-8") as f:
        fields = f.readline().strip().split(" ")
        total, dim = int(fields[0]), int(fields[1])
        for index in range(total):
            line = f.readline().strip()
            word, vec = line.rstrip().split(' ', 1)
            vec = np.array(vec.split(), dtype=np.float32)
            word_idx[word] = len(word_vectors)
            word_vectors.append(vec)
    unk_vec = np.random.uniform(-0.05, 0.05, word_vectors[0].shape).astype(np.float32)
    word_vectors.append(unk_vec)
    return np.asarray(word_vectors, dtype=np.float32), word_idx