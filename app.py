"""
Parth.ai — Vector Database Engine (Python Port of main.cpp)
Implements HNSW, KD-Tree, and Brute Force side-by-side with a RAG pipeline via Ollama.
"""

import math
import random
import threading
import time
import json
import re
from collections import defaultdict
from typing import Callable, List, Optional, Tuple, Dict
from flask import Flask, request, jsonify, send_file
import requests as http_requests

DIMS = 16  # demo vector dimensions

# ====================================================================
#  DISTANCE METRICS
# ====================================================================

def euclidean(a: List[float], b: List[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (na * nb)


def manhattan(a: List[float], b: List[float]) -> float:
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(metric: str) -> Callable:
    if metric == "cosine":
        return cosine
    if metric == "manhattan":
        return manhattan
    return euclidean


# ====================================================================
#  DATA TYPES
# ====================================================================

class VectorItem:
    def __init__(self, id: int, metadata: str, category: str, emb: List[float]):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb


# ====================================================================
#  BRUTE FORCE
# ====================================================================

class BruteForce:
    def __init__(self):
        self.items: List[VectorItem] = []

    def insert(self, v: VectorItem):
        self.items.append(v)

    def knn(self, q: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        results = [(dist_fn(q, v.emb), v.id) for v in self.items]
        results.sort(key=lambda x: x[0])
        return results[:k]

    def remove(self, id: int):
        self.items = [v for v in self.items if v.id != id]


# ====================================================================
#  KD-TREE
# ====================================================================

class KDNode:
    def __init__(self, item: VectorItem):
        self.item = item
        self.left: Optional["KDNode"] = None
        self.right: Optional["KDNode"] = None


class KDTree:
    def __init__(self, dims: int):
        self.dims = dims
        self.root: Optional[KDNode] = None

    def _insert(self, node: Optional[KDNode], item: VectorItem, depth: int) -> KDNode:
        if node is None:
            return KDNode(item)
        ax = depth % self.dims
        if item.emb[ax] < node.item.emb[ax]:
            node.left = self._insert(node.left, item, depth + 1)
        else:
            node.right = self._insert(node.right, item, depth + 1)
        return node

    def insert(self, v: VectorItem):
        self.root = self._insert(self.root, v, 0)

    def _knn(self, node: Optional[KDNode], q: List[float], k: int, depth: int,
             dist_fn: Callable, heap: List[Tuple[float, int]]):
        if node is None:
            return
        dn = dist_fn(q, node.item.emb)
        if len(heap) < k or dn < heap[0][0]:
            heap.append((dn, node.item.id))
            heap.sort(key=lambda x: -x[0])  # max-heap by negating for removal
            if len(heap) > k:
                heap.pop(0)

        ax = depth % self.dims
        diff = q[ax] - node.item.emb[ax]
        closer = node.left if diff < 0 else node.right
        farther = node.right if diff < 0 else node.left

        self._knn(closer, q, k, depth + 1, dist_fn, heap)
        if len(heap) < k or abs(diff) < heap[0][0]:
            self._knn(farther, q, k, depth + 1, dist_fn, heap)

    def knn(self, q: List[float], k: int, dist_fn: Callable) -> List[Tuple[float, int]]:
        heap: List[Tuple[float, int]] = []
        self._knn(self.root, q, k, 0, dist_fn, heap)
        heap.sort(key=lambda x: x[0])
        return heap

    def rebuild(self, items: List[VectorItem]):
        self.root = None
        for v in items:
            self.insert(v)


# ====================================================================
#  HNSW — Hierarchical Navigable Small World
# ====================================================================

class HNSW:
    class Node:
        def __init__(self, item: VectorItem, max_lyr: int):
            self.item = item
            self.max_lyr = max_lyr
            self.nbrs: List[List[int]] = [[] for _ in range(max_lyr + 1)]

    def __init__(self, M: int = 16, ef_build: int = 200):
        self.M = M
        self.M0 = 2 * M
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(M)
        self.G: Dict[int, HNSW.Node] = {}
        self.top_layer = -1
        self.entry_pt = -1
        self._rng = random.Random(42)

    def _rand_level(self) -> int:
        return int(math.floor(-math.log(self._rng.random()) * self.mL))

    def _search_layer(self, q: List[float], ep: int, ef: int, lyr: int,
                      dist_fn: Callable) -> List[Tuple[float, int]]:
        visited = {ep}
        d0 = dist_fn(q, self.G[ep].item.emb)
        # min-heap for candidates, max-heap for found
        cands = [(d0, ep)]
        found = [(d0, ep)]

        while cands:
            cands.sort(key=lambda x: x[0])
            cd, cid = cands.pop(0)
            if len(found) >= ef and cd > max(found, key=lambda x: x[0])[0]:
                break
            if lyr >= len(self.G[cid].nbrs):
                continue
            for nid in self.G[cid].nbrs[lyr]:
                if nid in visited or nid not in self.G:
                    continue
                visited.add(nid)
                nd = dist_fn(q, self.G[nid].item.emb)
                worst = max(found, key=lambda x: x[0])[0] if found else float('inf')
                if len(found) < ef or nd < worst:
                    cands.append((nd, nid))
                    found.append((nd, nid))
                    if len(found) > ef:
                        found.sort(key=lambda x: x[0])
                        found = found[:ef]

        found.sort(key=lambda x: x[0])
        return found

    def _select_nbrs(self, cands: List[Tuple[float, int]], max_m: int) -> List[int]:
        return [c[1] for c in cands[:max_m]]

    def insert(self, item: VectorItem, dist_fn: Callable):
        id = item.id
        lvl = self._rand_level()
        self.G[id] = HNSW.Node(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist_fn)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)

            # Extend nbrs list if needed
            while len(self.G[id].nbrs) <= lc:
                self.G[id].nbrs.append([])
            self.G[id].nbrs[lc] = sel

            for nid in sel:
                if nid not in self.G:
                    continue
                while len(self.G[nid].nbrs) <= lc:
                    self.G[nid].nbrs.append([])
                conn = self.G[nid].nbrs[lc]
                conn.append(id)
                if len(conn) > max_m:
                    ds = [(dist_fn(self.G[nid].item.emb, self.G[c].item.emb), c)
                          for c in conn if c in self.G]
                    ds.sort(key=lambda x: x[0])
                    self.G[nid].nbrs[lc] = [d[1] for d in ds[:max_m]]

            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id

    def knn(self, q: List[float], k: int, ef: int,
            dist_fn: Callable) -> List[Tuple[float, int]]:
        if self.entry_pt == -1:
            return []
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(self.G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist_fn)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist_fn)
        return W[:k]

    def remove(self, id: int):
        if id not in self.G:
            return
        for nid, nd in self.G.items():
            for layer in nd.nbrs:
                if id in layer:
                    layer.remove(id)
        if self.entry_pt == id:
            self.entry_pt = -1
            for nid in self.G:
                if nid != id:
                    self.entry_pt = nid
                    break
        del self.G[id]

    def get_info(self) -> dict:
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes = []
        edges = []

        for id, nd in self.G.items():
            nodes.append({
                "id": id,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr
            })
            for lc in range(min(nd.max_lyr + 1, max_l)):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id, "dst": nid, "lyr": lc})

        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges
        }

    def size(self) -> int:
        return len(self.G)


# ====================================================================
#  VECTOR DATABASE (16D demo index)
# ====================================================================

class VectorDB:
    def __init__(self, dims: int):
        self.dims = dims
        self.store: Dict[int, VectorItem] = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self._lock = threading.Lock()
        self._next_id = 1

    def insert(self, metadata: str, category: str, emb: List[float],
               dist_fn: Callable) -> int:
        with self._lock:
            v = VectorItem(self._next_id, metadata, category, emb)
            self._next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist_fn)
            return v.id

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.bf.remove(id)
            self.hnsw.remove(id)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q: List[float], k: int, metric: str,
               algo: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter_ns()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = (time.perf_counter_ns() - t0) // 1000

            hits = []
            for d, id in raw:
                if id in self.store:
                    v = self.store[id]
                    hits.append({
                        "id": v.id,
                        "metadata": v.metadata,
                        "category": v.category,
                        "distance": round(d, 6),
                        "embedding": [round(x, 4) for x in v.emb]
                    })
            return {"results": hits, "latencyUs": us, "algo": algo, "metric": metric}

    def benchmark(self, q: List[float], k: int, metric: str) -> dict:
        with self._lock:
            dfn = get_dist_fn(metric)

            t0 = time.perf_counter_ns()
            self.bf.knn(q, k, dfn)
            bf_us = (time.perf_counter_ns() - t0) // 1000

            t0 = time.perf_counter_ns()
            self.kdt.knn(q, k, dfn)
            kd_us = (time.perf_counter_ns() - t0) // 1000

            t0 = time.perf_counter_ns()
            self.hnsw.knn(q, k, 50, dfn)
            hnsw_us = (time.perf_counter_ns() - t0) // 1000

            return {
                "bruteforceUs": bf_us,
                "kdtreeUs": kd_us,
                "hnswUs": hnsw_us,
                "itemCount": len(self.store)
            }

    def all(self) -> List[dict]:
        with self._lock:
            return [
                {
                    "id": v.id,
                    "metadata": v.metadata,
                    "category": v.category,
                    "embedding": [round(x, 4) for x in v.emb]
                }
                for v in self.store.values()
            ]

    def hnsw_info(self) -> dict:
        with self._lock:
            return self.hnsw.get_info()

    def size(self) -> int:
        with self._lock:
            return len(self.store)


# ====================================================================
#  TEXT CHUNKER
# ====================================================================

def chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30) -> List[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunk = " ".join(words[i:end])
        chunks.append(chunk)
        if end == len(words):
            break
        i += step
    return chunks


# ====================================================================
#  OLLAMA CLIENT
# ====================================================================

class OllamaClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2:1b"

    def is_available(self) -> bool:
        try:
            r = http_requests.get(f"{self.base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def available_models(self) -> List[str]:
        """Return list of locally downloaded model names."""
        try:
            r = http_requests.get(f"{self.base}/api/tags", timeout=2)
            if r.status_code != 200:
                return []
            return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    def is_gen_model_ready(self) -> bool:
        """Check if the generation model is actually downloaded."""
        return any(self.gen_model in m for m in self.available_models())

    def embed(self, text: str) -> List[float]:
        try:
            r = http_requests.post(
                f"{self.base}/api/embeddings",
                json={"model": self.embed_model, "prompt": text},
                timeout=30
            )
            if r.status_code != 200:
                return []
            data = r.json()
            return data.get("embedding", [])
        except Exception:
            return []

    def generate(self, prompt: str) -> str:
        try:
            r = http_requests.post(
                f"{self.base}/api/generate",
                json={"model": self.gen_model, "prompt": prompt, "stream": False},
                timeout=180
            )
            if r.status_code == 404:
                return (f"ERROR: Model '{self.gen_model}' not found. "
                        f"It may still be downloading. Run: ollama pull {self.gen_model}")
            if r.status_code != 200:
                return f"ERROR: Ollama returned status {r.status_code}. Run: ollama serve"
            return r.json().get("response", "")
        except http_requests.exceptions.ConnectionError:
            return "ERROR: Cannot connect to Ollama. Run: ollama serve"
        except Exception as e:
            return f"ERROR: {str(e)}"


# ====================================================================
#  DOCUMENT DATABASE
# ====================================================================

class DocItem:
    def __init__(self, id: int, title: str, text: str, emb: List[float]):
        self.id = id
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self.store: Dict[int, DocItem] = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self._lock = threading.Lock()
        self._next_id = 1
        self._dims = 0

    def insert(self, title: str, text: str, emb: List[float]) -> int:
        with self._lock:
            if self._dims == 0:
                self._dims = len(emb)
            item = DocItem(self._next_id, title, text, emb)
            self._next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q: List[float], k: int,
               max_dist: float = 0.7) -> List[Tuple[float, "DocItem"]]:
        with self._lock:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[id]) for d, id in raw
                    if id in self.store and d <= max_dist]

    def remove(self, id: int) -> bool:
        with self._lock:
            if id not in self.store:
                return False
            del self.store[id]
            self.hnsw.remove(id)
            self.bf.remove(id)
            return True

    def all(self) -> List[DocItem]:
        with self._lock:
            return list(self.store.values())

    def size(self) -> int:
        with self._lock:
            return len(self.store)

    def get_dims(self) -> int:
        return self._dims


# ====================================================================
#  DEMO DATA (16D categorical vectors)
# ====================================================================

DEMO_DATA = [
    ("Linked List: nodes connected by pointers", "cs",
     [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06]),
    ("Binary Search Tree: O(log n) search and insert", "cs",
     [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10]),
    ("Dynamic Programming: memoization overlapping subproblems", "cs",
     [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07]),
    ("Graph BFS and DFS: breadth and depth first traversal", "cs",
     [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07]),
    ("Hash Table: O(1) lookup with collision chaining", "cs",
     [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09]),
    ("Calculus: derivatives integrals and limits", "math",
     [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10]),
    ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
     [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09]),
    ("Probability: distributions random variables Bayes theorem", "math",
     [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08]),
    ("Number Theory: primes modular arithmetic RSA cryptography", "math",
     [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06]),
    ("Combinatorics: permutations combinations generating functions", "math",
     [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07]),
    ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07]),
    ("Sushi: vinegared rice raw fish and nori rolls", "food",
     [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08]),
    ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
     [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06]),
    ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
     [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09]),
    ("Croissant: laminated pastry with buttery flaky layers", "food",
     [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06]),
    ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
     [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72]),
    ("Football: tackles touchdowns field goals and strategy", "sports",
     [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76]),
    ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
     [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82]),
    ("Chess: openings endgames tactics strategic board game", "sports",
     [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90]),
    ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
     [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80]),
]


def load_demo(db: VectorDB):
    dist = get_dist_fn("cosine")
    for meta, cat, emb in DEMO_DATA:
        db.insert(meta, cat, emb, dist)


# ====================================================================
#  FLASK APPLICATION
# ====================================================================

app = Flask(__name__, static_folder=".", static_url_path="")

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


@app.after_request
def add_cors(response):
    return cors_headers(response)


@app.route("/", methods=["GET"])
def index():
    return send_file("index.html")


# ── DEMO VECTOR ENDPOINTS ─────────────────────────────────────────────

@app.route("/search", methods=["GET"])
def search():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400

    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    algo = request.args.get("algo", "hnsw")

    result = db.search(q, k, metric, algo)
    return jsonify(result)


@app.route("/insert", methods=["POST"])
def insert():
    data = request.get_json(silent=True) or {}
    meta = data.get("metadata", "")
    cat = data.get("category", "")
    emb = data.get("embedding", [])
    if not meta or not emb or len(emb) != DIMS:
        return jsonify({"error": "invalid body"}), 400
    id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
    return jsonify({"id": id})


@app.route("/delete/<int:id>", methods=["DELETE"])
def delete(id: int):
    ok = db.remove(id)
    return jsonify({"ok": ok})


@app.route("/items", methods=["GET"])
def items():
    return jsonify(db.all())


@app.route("/benchmark", methods=["GET"])
def benchmark():
    v_str = request.args.get("v", "")
    try:
        q = [float(x) for x in v_str.split(",") if x.strip()]
    except ValueError:
        q = []
    if len(q) != DIMS:
        return jsonify({"error": f"need {DIMS}D vector"}), 400
    k = int(request.args.get("k", 5))
    metric = request.args.get("metric", "cosine")
    return jsonify(db.benchmark(q, k, metric))


@app.route("/hnsw-info", methods=["GET"])
def hnsw_info():
    return jsonify(db.hnsw_info())


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "count": db.size(),
        "dims": DIMS,
        "algorithms": ["bruteforce", "kdtree", "hnsw"],
        "metrics": ["euclidean", "cosine", "manhattan"]
    })


# ── DOCUMENT + RAG ENDPOINTS ─────────────────────────────────────────

@app.route("/doc/insert", methods=["POST"])
def doc_insert():
    data = request.get_json(silent=True) or {}
    title = data.get("title", "").strip()
    text = data.get("text", "").strip()
    if not title or not text:
        return jsonify({"error": "need title and text"}), 400

    chunks = chunk_text(text, 250, 30)
    ids = []
    for i, chunk in enumerate(chunks):
        emb = ollama.embed(chunk)
        if not emb:
            return jsonify({
                "error": "Ollama unavailable. Install from https://ollama.com then run: "
                         "ollama pull nomic-embed-text && ollama pull llama3.2"
            }), 503
        chunk_title = f"{title} [{i+1}/{len(chunks)}]" if len(chunks) > 1 else title
        ids.append(doc_db.insert(chunk_title, chunk, emb))

    return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.get_dims()})


@app.route("/doc/delete/<int:id>", methods=["DELETE"])
def doc_delete(id: int):
    ok = doc_db.remove(id)
    return jsonify({"ok": ok})


@app.route("/doc/list", methods=["GET"])
def doc_list():
    docs = doc_db.all()
    result = []
    for d in docs:
        preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
        word_count = len(d.text.split())
        result.append({
            "id": d.id,
            "title": d.title,
            "preview": preview,
            "words": word_count
        })
    return jsonify(result)


@app.route("/doc/search", methods=["POST"])
def doc_search():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    k = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama unavailable"}), 503

    hits = doc_db.search(q_emb, k)
    return jsonify({
        "contexts": [
            {"id": h.id, "title": h.title, "distance": round(d, 4)}
            for d, h in hits
        ]
    })


@app.route("/doc/ask", methods=["POST"])
def doc_ask():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    k = int(data.get("k", 3))
    if not question:
        return jsonify({"error": "need question"}), 400

    # Step 1: embed question
    q_emb = ollama.embed(question)
    if not q_emb:
        return jsonify({"error": "Ollama embed model unavailable. Run: ollama pull nomic-embed-text"}), 503

    # Check gen model is ready before searching/generating
    if not ollama.is_gen_model_ready():
        return jsonify({
            "error": f"LLM model '{ollama.gen_model}' is not yet downloaded. "
                     f"Please wait for: ollama pull {ollama.gen_model} to complete."
        }), 503

    # Step 2: retrieve context
    hits = doc_db.search(q_emb, k)

    # Step 3: build prompt
    ctx_parts = []
    for i, (d, h) in enumerate(hits):
        ctx_parts.append(f"[{i+1}] {h.title}:\n{h.text}\n\n")
    ctx_str = "".join(ctx_parts)

    prompt = (
        "You are a helpful assistant. Answer the user's question directly. "
        "Use the provided context if it contains relevant information. "
        "If it doesn't, just use your own general knowledge. "
        "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
        "'the context doesn't mention'. Just answer the question naturally.\n\n"
        f"Context:\n{ctx_str}"
        f"Question: {question}\n\nAnswer:"
    )

    # Step 4: generate
    answer = ollama.generate(prompt)

    return jsonify({
        "answer": answer,
        "model": ollama.gen_model,
        "contexts": [
            {"id": h.id, "title": h.title, "text": h.text, "distance": round(d, 4)}
            for d, h in hits
        ],
        "docCount": doc_db.size()
    })


@app.route("/status", methods=["GET"])
def status():
    up = ollama.is_available()
    gen_ready = ollama.is_gen_model_ready() if up else False
    return jsonify({
        "ollamaAvailable": up,
        "genModelReady": gen_ready,
        "embedModel": ollama.embed_model,
        "genModel": ollama.gen_model,
        "availableModels": ollama.available_models() if up else [],
        "docCount": doc_db.size(),
        "docDims": doc_db.get_dims(),
        "demoDims": DIMS,
        "demoCount": db.size()
    })


# ── CORS PREFLIGHT ────────────────────────────────────────────────────

@app.route("/", methods=["OPTIONS"])
@app.route("/<path:path>", methods=["OPTIONS"])
def options(path=""):
    from flask import Response
    resp = Response(status=204)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


# ====================================================================
#  STARTUP
# ====================================================================

if __name__ == "__main__":
    load_demo(db)
    ollama_up = ollama.is_available()
    print("=== Parth.ai Vector Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print(f"Ollama: {'ONLINE' if ollama_up else 'OFFLINE (install from ollama.com)'}")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")
    app.run(host="0.0.0.0", port=8080, debug=False)
