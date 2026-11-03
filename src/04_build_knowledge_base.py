"""Task 3 (part 2) - build the knowledge base + retrieval index.  Run: python src/04_build_knowledge_base.py"""
import json

from common import OUT
from rag import KnowledgeBase

kb = KnowledgeBase.build()
by_type = {}
for d in kb.docs:
    by_type[d["type"]] = by_type.get(d["type"], 0) + 1
print("Knowledge base documents:", by_type, "total", len(kb.docs))
(OUT / "knowledge_base_docs.json").write_text(json.dumps(kb.docs, indent=1, default=str))

# quick retrieval sanity checks
for q in ["database connection pool exhausted", "NEFT failures gateway timeout", "CPU 94% memory 89% DB pool 93%", "rate limit"]:
    print(f"\nQ: {q}")
    for h in kb.search(q, 3):
        print(f"   {h['score']:.2f}  [{h['type']}] {h['id']} {h['title']}")
