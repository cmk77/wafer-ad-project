"""ontology.py — 웨이퍼 결함 온톨로지 로드/질의.

JSON 그래프(시드)를 로드해 결함 유형 → 공정/원인/조치 서브그래프를 조회하고,
Gemma 프롬프트에 넣을 한국어 컨텍스트 블록을 생성한다.
확장 시 to_networkx()로 그래프 분석, rdflib로 OWL 이관 가능.
"""
from __future__ import annotations

import difflib
import json
from pathlib import Path


NORMAL_LABELS = {"none", "normal", "good", "정상"}


class WaferOntology:
    def __init__(self, path: str | Path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.defects: dict = data["defect_types"]
        self.steps: dict = data["process_steps"]
        # alias(자유 표기) → 정식 키. 각 항목의 "aliases" 필드에서 구축.
        self.aliases: dict[str, str] = {}
        for k, d in self.defects.items():
            for al in d.get("aliases", []):
                self.aliases[al.lower().replace("-", "_").replace(" ", "_")] = k

    # ------------------------------------------------------------- query
    def resolve(self, name: str) -> str | None:
        """레지스트리 라벨(자유 표기)을 온톨로지 키로 정규화.

        우선순위: 정식 키 → 명시적 alias → difflib 유사 매칭(0.6).
        'none' 등 정상 라벨은 결함이 아니므로 None (context_block이 별도 처리).
        """
        key = name.lower().replace("-", "_").replace(" ", "_")
        if key in NORMAL_LABELS:
            return None
        if key in self.defects:
            return key
        if key in self.aliases:
            return self.aliases[key]
        cand = difflib.get_close_matches(key, self.defects.keys(), n=1, cutoff=0.6)
        return cand[0] if cand else None

    def subgraph(self, defect: str) -> dict | None:
        key = self.resolve(defect)
        if key is None:
            return None
        d = self.defects[key]
        return {
            "defect": key,
            "label_ko": d["label_ko"],
            "signature": d["signature"],
            "severity": d["severity"],
            "process_steps": [
                {"id": s, **self.steps.get(s, {})} for s in d["process_steps"]
            ],
            "root_causes": d["root_causes"],
            "actions": d["actions"],
            "related": d.get("related", []),
        }

    # ------------------------------------------------------- LLM context
    def context_block(self, defect: str) -> str:
        """Gemma 프롬프트에 삽입할 grounding 컨텍스트 (한국어)."""
        if defect.lower().replace("-", "_") in NORMAL_LABELS:
            return "[온톨로지] 정상 판정 — 결함 원인 분석 대상이 아닙니다."
        g = self.subgraph(defect)
        if g is None:
            return (f"[온톨로지] '{defect}' 는 미등록 유형입니다. "
                    "아래 리포트에서 원인 추정은 반드시 '추정'으로 표기하고, "
                    "온톨로지 등록을 권고하세요.")
        steps = ", ".join(f"{s['label_ko']}({s['id']})" for s in g["process_steps"])
        return (
            f"[온톨로지: {g['label_ko']} ({g['defect']}), 심각도 {g['severity']}]\n"
            f"- 시그니처: {g['signature']}\n"
            f"- 연관 공정: {steps}\n"
            f"- 원인 후보: {', '.join(g['root_causes'])}\n"
            f"- 권장 조치: {', '.join(g['actions'])}\n"
            f"- 유사 유형: {', '.join(g['related']) or '없음'}"
        )

    # --------------------------------------------------------- extension
    def to_networkx(self):
        import networkx as nx
        G = nx.DiGraph()
        for k, d in self.defects.items():
            G.add_node(k, kind="defect", **{"label_ko": d["label_ko"]})
            for s in d["process_steps"]:
                G.add_node(s, kind="process")
                G.add_edge(k, s, rel="occurs_in")
            for rc in d["root_causes"]:
                G.add_node(rc, kind="cause")
                G.add_edge(k, rc, rel="caused_by")
            for a in d["actions"]:
                G.add_node(a, kind="action")
                G.add_edge(k, a, rel="mitigated_by")
            for r in d.get("related", []):
                G.add_edge(k, r, rel="related_to")
        return G


if __name__ == "__main__":
    import sys
    onto = WaferOntology(sys.argv[1] if len(sys.argv) > 1
                         else "data/ontology/wafer_defect_ontology.json")
    print(onto.context_block(sys.argv[2] if len(sys.argv) > 2 else "scratch"))
