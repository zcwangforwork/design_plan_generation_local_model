"""
VectorStore - ChromaDB 向量存储封装

提供文档摄入和语义检索功能。
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Optional, List, Dict

import chromadb
from chromadb.config import Settings

from app.services.rag.embedder import Embedder
from app.services.rag.tokenizer import get_tokenizer

# 延迟导入
_chroma_client = None


class VectorStore:
    """ChromaDB 向量存储封装"""

    # ChromaDB 持久化目录
    BASE_DIR = Path(__file__).parent.parent.parent.parent / "chroma_db_insulin_pump"
    COLLECTION_NAME_PREFIX = "qms_doc_"

    # 查询时使用的目标 collection 列表（优先级从高到低）
    QUERY_COLLECTIONS = ["insulin_pump_kb"]

    # 额外的 ChromaDB 目录及其 collection 列表
    # 格式: {db_path: [collection_names]}
    EXTRA_DB_CONFIG = {}

    # BM25 索引缓存: collection_name → (chunk_count, BM25Okapi_instance, tokenized_corpus)
    _bm25_cache: Dict[str, tuple] = {}
    _bm25_cache_lock = threading.Lock()

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None
    ):
        """
        初始化向量存储

        Args:
            persist_directory: ChromaDB 数据持久化目录
            collection_name: collection 名称（不包含前缀），默认使用 "all"
        """
        self.persist_directory = persist_directory or str(self.BASE_DIR)
        self.collection_name = collection_name or "all"

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        self._client = None
        self._collection = None
        self._embedder = None

    @property
    def client(self):
        """延迟初始化 ChromaDB 客户端"""
        global _chroma_client
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        return _chroma_client

    # 额外客户端缓存: {path: client}
    _extra_clients = {}

    @classmethod
    def _get_client_for_path(cls, db_path: str):
        """获取或创建指定路径的 ChromaDB 客户端"""
        db_path = os.path.abspath(db_path)
        if db_path not in cls._extra_clients:
            os.makedirs(db_path, exist_ok=True)
            cls._extra_clients[db_path] = chromadb.PersistentClient(
                path=db_path,
                settings=Settings(anonymized_telemetry=False)
            )
        return cls._extra_clients[db_path]

    @classmethod
    def set_extra_db_config(cls, config: dict):
        """
        设置额外的 ChromaDB 目录配置

        Args:
            config: {db_path: [collection_names]}
        """
        cls.EXTRA_DB_CONFIG = config
        # 清除旧的额外客户端缓存
        cls._extra_clients = {}

    @property
    def embedder(self):
        """延迟初始化嵌入器"""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def collection(self):
        """获取或创建 collection"""
        if self._collection is None:
            full_name = f"{self.COLLECTION_NAME_PREFIX}{self.collection_name}"
            self._collection = self.client.get_or_create_collection(
                name=full_name,
                metadata={"description": "QMS 医疗器械文档参考库"}
            )
        return self._collection

    def add_chunk(
        self,
        chunk_id: str,
        text: str,
        doc_type: str,
        source_file: str,
        section_title: Optional[str] = None,
        chunk_index: int = 0,
        metadata: Optional[dict] = None
    ):
        """
        添加单个文档块到向量库

        Args:
            chunk_id: 块唯一 ID
            text: 文本内容
            doc_type: 文档类型（用于过滤）
            source_file: 来源文件名
            section_title: 章节标题
            chunk_index: 块在文档中的顺序索引
            metadata: 额外元数据
        """
        meta = {
            "doc_type": doc_type,
            "source_file": source_file,
            "section_title": section_title or "",
            "chunk_index": chunk_index,
            **(metadata or {})
        }

        # 使用我们的嵌入器生成向量
        embedding = self.embedder.encode_single(text)

        self.collection.add(
            ids=[chunk_id],
            documents=[text],
            metadatas=[meta],
            embeddings=[embedding]
        )

    def add_chunks(self, chunks: list[dict]):
        """
        批量添加文档块

        Args:
            chunks: 块列表，每项包含 chunk_id, text, doc_type, source_file 等
        """
        if not chunks:
            return

        ids = [c["chunk_id"] for c in chunks]
        texts = [c["text"] for c in chunks]
        metadatas = [
            {
                "doc_type": c.get("doc_type", ""),
                "source_file": c.get("source_file", ""),
                "section_title": c.get("section_title", ""),
                "chunk_index": c.get("chunk_index", 0)
            }
            for c in chunks
        ]

        # 使用我们的嵌入器批量生成向量
        embeddings = self.embedder.encode(texts)

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def retrieve(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.0  # 默认 0.0 允许几乎所有结果通过
    ) -> list[dict]:
        """
        语义检索 - 从多个 collection 检索并合并结果

        Args:
            query: 查询文本（产品信息）
            doc_type: 文档类型过滤（可选）
            top_k: 返回数量
            similarity_threshold: 最低相似度阈值（0-1），默认 0.0

        Returns:
            检索结果列表，每项包含 text, source_file, section_title, distance
        """
        query_embedding = self.embedder.encode_single(query)
        all_results = []

        # 构建所有 (client, collection_name) 查询对
        query_targets = [(self.client, name) for name in self.QUERY_COLLECTIONS]
        for db_path, coll_names in self.EXTRA_DB_CONFIG.items():
            client = self._get_client_for_path(db_path)
            for name in coll_names:
                query_targets.append((client, name))

        for client, coll_name in query_targets:
            try:
                coll = client.get_collection(name=coll_name)
                results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"]
                )
                if results["ids"]:
                    for i in range(len(results["ids"][0])):
                        distance = results["distances"][0][i]
                        similarity = max(0.0, 1.0 - distance / 2.0)
                        if similarity < similarity_threshold:
                            continue
                        meta = results["metadatas"][0][i] or {}
                        # doc_type 过滤
                        if doc_type and meta.get("doc_type", "") != doc_type:
                            continue
                        all_results.append({
                            "text": results["documents"][0][i],
                            "source_file": meta.get("source_file", ""),
                            "section_title": meta.get("section_title", ""),
                            "doc_type": meta.get("doc_type", ""),
                            "chunk_index": meta.get("chunk_index", 0),
                            "similarity": similarity,
                            "distance": distance
                        })
            except Exception as e:
                print(f"    [VectorStore] 查询 collection '{coll_name}' 失败: {e}")
                continue

        # 按相似度排序并返回 top_k
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        return all_results[:top_k]

    def retrieve_hybrid(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.3,
        vector_weight: float = 0.85
    ) -> list[dict]:
        """
        混合检索：从多个 collection 语义向量检索

        注意：当 collection 数据量过大时（>5万条），BM25 检索需加载全量数据到内存，
        开销巨大，因此对大 collection 仅使用向量检索。

        Args:
            query: 查询文本
            doc_type: 文档类型过滤（可选），如果过滤后无结果则忽略
            top_k: 返回数量
            similarity_threshold: 最低相似度阈值（0-1）
            vector_weight: 向量检索权重 (0-1)，BM25权重 = 1 - vector_weight

        Returns:
            合并后的检索结果列表，按综合分数排序
        """
        t_start = time.time()
        print(f"    [VectorStore] retrieve_hybrid start — query='{query[:80]}'")

        t_embed_start = time.time()
        query_embedding = self.embedder.encode_single(query)
        print(f"    [VectorStore] embedding done ({time.time() - t_embed_start:.1f}s)")

        # 构建所有 (client, collection_name) 查询对
        query_targets = [(self.client, name) for name in self.QUERY_COLLECTIONS]
        for db_path, coll_names in self.EXTRA_DB_CONFIG.items():
            client = self._get_client_for_path(db_path)
            for name in coll_names:
                query_targets.append((client, name))

        # 1. 向量检索 - 从所有 collection 检索并合并
        vector_dict = {}
        for client, coll_name in query_targets:
            try:
                coll = client.get_collection(name=coll_name)
                vector_results = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k * 2,
                    include=["documents", "metadatas", "distances"]
                )
                if vector_results["ids"]:
                    for i in range(len(vector_results["ids"][0])):
                        chunk_id = vector_results["ids"][0][i]
                        distance = vector_results["distances"][0][i]
                        similarity = max(0.0, 1.0 - distance / 2.0)

                        if similarity >= similarity_threshold:
                            meta = vector_results["metadatas"][0][i] or {}
                            chunk_doc_type = meta.get("doc_type", "")

                            # doc_type 过滤
                            if doc_type and chunk_doc_type != doc_type:
                                continue

                            # 去重：同一 chunk_id 只保留相似度更高的
                            if chunk_id not in vector_dict or similarity > vector_dict[chunk_id]["vector_score"]:
                                vector_dict[chunk_id] = {
                                    "text": vector_results["documents"][0][i],
                                    "source_file": meta.get("source_file", ""),
                                    "section_title": meta.get("section_title", ""),
                                    "doc_type": chunk_doc_type,
                                    "chunk_index": meta.get("chunk_index", 0),
                                    "similarity": similarity,
                                    "vector_score": similarity,
                                    "bm25_score": 0.0
                                }
            except Exception as e:
                print(f"    [VectorStore] 混合检索 collection '{coll_name}' 失败: {e}")
                continue

        # 如果 doc_type 过滤后无结果，忽略过滤重新检索
        if not vector_dict and doc_type:
            print(f"    [RAG] doc_type='{doc_type}' 过滤无结果，忽略类型过滤")
            for client, coll_name in query_targets:
                try:
                    coll = client.get_collection(name=coll_name)
                    vector_results = coll.query(
                        query_embeddings=[query_embedding],
                        n_results=top_k * 2,
                        include=["documents", "metadatas", "distances"]
                    )
                    if vector_results["ids"]:
                        for i in range(len(vector_results["ids"][0])):
                            chunk_id = vector_results["ids"][0][i]
                            distance = vector_results["distances"][0][i]
                            similarity = max(0.0, 1.0 - distance / 2.0)

                            if similarity >= similarity_threshold:
                                meta = vector_results["metadatas"][0][i] or {}
                                if chunk_id not in vector_dict or similarity > vector_dict[chunk_id]["vector_score"]:
                                    vector_dict[chunk_id] = {
                                        "text": vector_results["documents"][0][i],
                                        "source_file": meta.get("source_file", ""),
                                        "section_title": meta.get("section_title", ""),
                                        "doc_type": meta.get("doc_type", ""),
                                        "chunk_index": meta.get("chunk_index", 0),
                                        "similarity": similarity,
                                        "vector_score": similarity,
                                        "bm25_score": 0.0
                                    }
                except Exception:
                    continue

        t_vector_done = time.time()
        print(f"    [VectorStore] vector search done — {len(vector_dict)} results ({t_vector_done - t_embed_start:.1f}s from embed start)")

        # 2. BM25 关键词检索 - 仅对小 collection 执行（<5万条）
        bm25_scores = {}
        for client, coll_name in query_targets:
            try:
                coll = client.get_collection(name=coll_name)
                coll_count = coll.count()
                if coll_count < 50000:
                    scores = self._bm25_search_collection(client, coll_name, query, None, top_k * 2)
                    for cid, score in scores.items():
                        if cid not in bm25_scores or score > bm25_scores[cid]:
                            bm25_scores[cid] = score
                else:
                    print(f"    [VectorStore] 跳过 BM25 检索（{coll_name} 有 {coll_count} 条，超过 5 万阈值）")
            except Exception:
                continue

        t_bm25_done = time.time()
        print(f"    [VectorStore] BM25 done — {len(bm25_scores)} results ({t_bm25_done - t_vector_done:.1f}s)")

        # 3. 合并结果
        all_ids = set(vector_dict.keys()) | set(bm25_scores.keys())

        merged = []
        for chunk_id in all_ids:
            vec_score = vector_dict.get(chunk_id, {}).get("vector_score", 0.0)
            bm_score = bm25_scores.get(chunk_id, 0.0)

            # 归一化并计算综合分数
            if vec_score > 0 and bm_score > 0:
                combined_score = vector_weight * vec_score + (1 - vector_weight) * bm_score
            elif vec_score > 0:
                combined_score = vec_score * vector_weight
            elif bm_score > 0:
                combined_score = bm_score * (1 - vector_weight)
            else:
                continue

            chunk_info = vector_dict.get(chunk_id, {})
            merged.append({
                "text": chunk_info.get("text", ""),
                "source_file": chunk_info.get("source_file", ""),
                "section_title": chunk_info.get("section_title", ""),
                "doc_type": chunk_info.get("doc_type", ""),
                "chunk_index": chunk_info.get("chunk_index", 0),
                "similarity": combined_score,
                "vector_score": vec_score,
                "bm25_score": bm_score
            })

        # 按综合分数排序
        merged.sort(key=lambda x: x["similarity"], reverse=True)

        print(f"    [VectorStore] retrieve_hybrid done — {len(merged[:top_k])} results, total {time.time() - t_start:.1f}s")

        return merged[:top_k]

    @classmethod
    def _get_or_build_bm25(cls, collection_name: str, docs: List[str]):
        """
        获取或构建缓存的 BM25Okapi 实例

        当 collection 的 chunk 数量不变时，复用已构建的 BM25 索引，
        避免每次查询都重新加载全量文档 + 重新分词。

        Returns:
            (BM25Okapi_instance, tokenized_corpus)
        """
        from rank_bm25 import BM25Okapi

        chunk_count = len(docs)

        with cls._bm25_cache_lock:
            cached = cls._bm25_cache.get(collection_name)
            if cached and cached[0] == chunk_count:
                return cached[1], cached[2]

            # 重建：分词 + 构建 BM25
            tokenizer = get_tokenizer()
            tokenized_corpus = tokenizer.tokenize_corpus(docs)
            bm25 = BM25Okapi(
                tokenized_corpus,
                k1=1.2,    # 降低高频词惩罚（中文法规文档核心术语频繁出现）
                b=0.6,     # 放宽长文档惩罚（中文 chunk 普遍较长）
                epsilon=0.25
            )

            # 缓存
            cls._bm25_cache[collection_name] = (chunk_count, bm25, tokenized_corpus)

            # LRU 淘汰：超过 10 个缓存项时删除最旧的
            if len(cls._bm25_cache) > 10:
                oldest_key = next(iter(cls._bm25_cache))
                del cls._bm25_cache[oldest_key]

            return bm25, tokenized_corpus

    @classmethod
    def invalidate_bm25_cache(cls, collection_name: str = None):
        """使 BM25 缓存失效（文档摄入后调用）"""
        with cls._bm25_cache_lock:
            if collection_name:
                cls._bm25_cache.pop(collection_name, None)
            else:
                cls._bm25_cache.clear()

    def _bm25_search(
        self,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 10
    ) -> dict:
        """
        BM25 关键词检索 - 从默认 collection 检索

        Returns:
            {chunk_id: bm25_score, ...}
        """
        full_name = f"{self.COLLECTION_NAME_PREFIX}{self.collection_name}"
        return self._bm25_search_collection(self.client, full_name, query, doc_type, top_k)

    def _bm25_search_collection(
        self,
        client,
        collection_name: str,
        query: str,
        doc_type: Optional[str] = None,
        top_k: int = 10
    ) -> dict:
        """
        BM25 关键词检索 - 从指定 collection 检索（jieba 中文分词版）

        改动点：
        1. str.split() → jieba.cut_for_search() 中文分词
        2. 增加 BM25 索引缓存，避免每次查询重新构建
        3. doc_type 过滤后若为空，直接返回空字典（避免无效分词）

        Args:
            client: ChromaDB 客户端
            collection_name: collection 名称
            query: 查询文本
            doc_type: 文档类型过滤
            top_k: 返回数量

        Returns:
            {chunk_id: bm25_score, ...}
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return {}

        try:
            coll = client.get_collection(name=collection_name)

            # 根据是否过滤选择加载方式
            if doc_type:
                result = coll.get(include=["documents", "metadatas"])

                if not result["ids"]:
                    return {}

                # doc_type 过滤
                ids, docs = [], []
                for i, meta in enumerate(result["metadatas"]):
                    if meta and meta.get("doc_type") == doc_type:
                        ids.append(result["ids"][i])
                        docs.append(result["documents"][i])

                if not docs:
                    return {}

                # 小批量手动构建 BM25（不缓存，因为过滤后的子集每次可能不同）
                tokenizer = get_tokenizer()
                tokenized_corpus = tokenizer.tokenize_corpus(docs)
                bm25 = BM25Okapi(
                    tokenized_corpus,
                    k1=1.2,
                    b=0.6,
                    epsilon=0.25
                )
            else:
                # 无过滤：使用缓存
                result = coll.get(include=["documents", "metadatas"])

                if not result["ids"]:
                    return {}

                ids = result["ids"]
                docs = result["documents"]

                bm25, _tokenized_corpus = self._get_or_build_bm25(collection_name, docs)

            # 查询分词
            tokenizer = get_tokenizer()
            query_tokens = list(tokenizer.tokenize(query))

            if not query_tokens:
                return {}

            scores = bm25.get_scores(query_tokens)

            # 返回 top_k
            top_indices = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:top_k]

            return {
                ids[i]: scores[i]
                for i in top_indices
                if scores[i] > 0
            }

        except Exception as e:
            print(f"    [VectorStore] BM25 检索失败: {e}")
            return {}

    def count(self) -> int:
        """返回所有查询 collection 中的文档块总数"""
        total = 0
        for coll_name in self.QUERY_COLLECTIONS:
            try:
                coll = self.client.get_collection(name=coll_name)
                total += coll.count()
            except Exception:
                continue
        for db_path, coll_names in self.EXTRA_DB_CONFIG.items():
            try:
                client = self._get_client_for_path(db_path)
                for coll_name in coll_names:
                    try:
                        coll = client.get_collection(name=coll_name)
                        total += coll.count()
                    except Exception:
                        continue
            except Exception:
                continue
        # 如果目标 collection 无数据，回退到默认 collection
        if total == 0:
            return self.collection.count()
        return total

    def clear(self):
        """清空 collection（谨慎使用）"""
        try:
            self.client.delete_collection(
                name=f"{self.COLLECTION_NAME_PREFIX}{self.collection_name}"
            )
            self._collection = None
        except Exception:
            pass

    def delete_by_source(self, source_file: str):
        """
        删除指定来源文件的所有块

        Args:
            source_file: 来源文件名
        """
        try:
            self.collection.delete(
                where={"source_file": source_file}
            )
        except Exception:
            pass

    def get_sources(self) -> list[str]:
        """获取向量库中所有来源文件列表"""
        try:
            result = self.collection.get(include=["metadatas"])
            sources = set()
            for meta in result.get("metadatas", []):
                if meta and meta.get("source_file"):
                    sources.add(meta["source_file"])
            return sorted(list(sources))
        except Exception:
            return []
