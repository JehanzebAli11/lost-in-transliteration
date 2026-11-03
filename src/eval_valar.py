"""RQ1/RQ2 for Valar, the author's own LLM traffic firewall (PLAN §4.2). Optional.

Runs only if either
  * env ``VALAR_API_URL`` and ``VALAR_API_KEY`` are set (HTTP mode), or
  * ``vendor/valar/valar_adapter.py`` exists and defines ``score_texts(list[str]) -> list[float]``
    returning a 0..1 injection score per text (in-process mode).
Otherwise it logs the skip, writes results/valar_status.json and exits 0, leaving a
placeholder the owner can fill later with ``make eval-valar``.

HTTP mode posts ``{<request_text_field>: text}`` with ``Authorization: Bearer <key>``
and reads the score / decision from the dotted paths in ``config.yaml: valar``
(TODO owner: set them to Valar's real request/response shape).

Output uses the same schema as the classifiers (results/classifiers/valar.jsonl,
family "valar"), so every table and figure includes it automatically. The write-up
must say Valar is the author's own system.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Callable

from .common import Config, load_config, load_dotenv_if_present, log_event, stage_parser, write_json, write_jsonl
from .data import load_all_items
from .eval_classifiers import eval_inputs, raw_row, write_summaries


def dotted(obj, path: str):
    cur = obj
    for part in path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        else:
            cur = cur[part]
    return cur


class ValarHTTP:
    def __init__(self, url: str, key: str, vc: dict, post: Callable | None = None):
        self.name = "valar"
        self.threshold = float(vc.get("threshold", 0.5))
        self.url, self.key, self.vc = url, key, vc
        if post is None:
            import httpx
            client = httpx.Client(timeout=60)

            def post(url, json, headers):
                r = client.post(url, json=json, headers=headers)
                r.raise_for_status()
                return r.json()
        self._post = post

    def score(self, texts: list[str]) -> list[float]:
        out = []
        for t in texts:
            resp = self._post(self.url, json={self.vc["request_text_field"]: t},
                              headers={"Authorization": f"Bearer {self.key}"})
            try:
                s = float(dotted(resp, self.vc["response_score_path"]))
            except (KeyError, IndexError, TypeError, ValueError):
                s = 1.0 if bool(dotted(resp, self.vc["response_flag_path"])) else 0.0
            out.append(s)
        return out


class ValarVendor:
    def __init__(self, path, threshold: float):
        spec = importlib.util.spec_from_file_location("valar_adapter", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self.name = "valar"
        self.threshold = threshold
        self._fn = mod.score_texts

    def score(self, texts):
        return [float(s) for s in self._fn(texts)]


def build_valar(cfg: Config):
    vc = cfg["valar"]
    url, key = os.environ.get(vc["url_env"]), os.environ.get(vc["key_env"])
    if url and key:
        return ValarHTTP(url, key, vc), "http"
    adapter = cfg.root / vc["vendor_dir"] / "valar_adapter.py"
    if adapter.exists():
        return ValarVendor(adapter, float(vc.get("threshold", 0.5))), "vendor"
    return None, "not configured"


def run(cfg: Config, *, system=None, limit: int | None = None) -> dict:
    mode = "injected"
    if system is None:
        system, mode = build_valar(cfg)
    status_path = cfg.results_dir / "valar_status.json"
    if system is None:
        msg = (f"Valar skipped: set {cfg['valar']['url_env']} + {cfg['valar']['key_env']} or add "
               f"{cfg['valar']['vendor_dir']}/valar_adapter.py, then run `make eval-valar`")
        write_json(status_path, {"status": "skipped", "reason": msg})
        log_event(cfg, msg, stage="valar")
        return {"status": "skipped"}
    items = load_all_items(cfg)
    if limit:
        items = items[:limit]
    triples = eval_inputs(cfg, items)
    scores = system.score([t for _, _, t in triples])
    rows = [raw_row("valar", "valar", it, kind, s, system.threshold)
            for (it, kind, _), s in zip(triples, scores)]
    write_jsonl(cfg.results_dir / "classifiers" / "valar.jsonl", rows)
    write_summaries(cfg)
    write_json(status_path, {"status": "done", "mode": mode, "n": len(rows),
                             "note": "Valar is the author's own system; say so in the write-up."})
    log_event(cfg, f"Valar ({mode}): scored {len(rows)} inputs", stage="valar")
    return {"status": "done", "n": len(rows)}


def main(argv=None):
    ap = stage_parser(__doc__)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.profile)
    load_dotenv_if_present(cfg)
    run(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
