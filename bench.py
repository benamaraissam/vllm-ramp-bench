#!/usr/bin/env python3
"""
Benchmark de montée en charge pour une API OpenAI-compatible (vLLM / Ray Serve LLM).

Il reproduit un trafic agentique de type « assistant de code » :
  - prompts longs (contexte de repo de 60k à 180k tokens par défaut) ;
  - sessions multi-tours sur le même contexte, donc prefix cache réaliste ;
  - sorties courtes (médiane ~200 tokens, moyenne ~600) ;
  - montée en charge par paliers d'utilisateurs simultanés, en boucle fermée.

Pour chaque palier il mesure TTFT, TPOT, latence totale, débit et taux
d'erreur, puis indique si les seuils (SLO) sont tenus.

Exemple :
    python bench.py --url http://127.0.0.1:8020/v1 --model qwen3.8-27b-fp8 \
        --steps 10,20,40,60,80,100,150 --step-duration 600
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

# --------------------------------------------------------------------------
# Texte synthétique déterministe : même graine -> même texte -> hit de cache
# --------------------------------------------------------------------------

_IDENTS = [
    "user", "order", "item", "cache", "client", "session", "token", "config",
    "result", "payload", "request", "response", "handler", "service", "store",
    "index", "buffer", "queue", "worker", "record", "account", "invoice",
]
_VERBS = ["get", "set", "load", "save", "parse", "build", "check", "send",
          "merge", "split", "find", "update", "delete", "create", "validate"]
_LINES = [
    "def {v}_{a}({b}, {c}):",
    "    {a} = {v}_{b}({c})",
    "    if {a} is None and {b} > {n}:",
    "        return {v}_{c}({a}, {b})",
    "    for {a} in {b}.{v}_{c}():",
    "        {a}.{v}({b}={n}, {c}=True)",
    "class {A}{B}Service:",
    "    # {v} the {a} before the {b} is sent to the {c}",
    "    raise ValueError(\"{a} {n} is not a valid {b}\")",
    "const {a}{B} = await {v}{A}({b}, {{ {c}: {n} }});",
    "",
]


def synth_text(seed: str, n_chars: int) -> str:
    """Pseudo-code déterministe d'environ n_chars caractères."""
    rnd = random.Random(seed)
    out: list[str] = []
    size = 0
    while size < n_chars:
        a, b, c = (rnd.choice(_IDENTS) for _ in range(3))
        line = rnd.choice(_LINES).format(
            a=a, b=b, c=c, v=rnd.choice(_VERBS), n=rnd.randint(0, 9999),
            A=a.capitalize(), B=b.capitalize(),
        )
        out.append(line)
        size += len(line) + 1
    return "\n".join(out)


# --------------------------------------------------------------------------
# Mesures
# --------------------------------------------------------------------------

@dataclass
class Sample:
    step_users: int
    user: int
    session: int
    turn: int
    started_s: float          # secondes depuis le début du palier
    inflight: int             # requêtes en vol au départ de celle-ci (elle incluse)
    prompt_tokens: int
    cached_tokens: int        # -1 si le serveur ne le renvoie pas
    completion_tokens: int
    ttft_s: float
    latency_s: float
    tpot_ms: float
    ok: bool
    error: str = ""


def pct(values: list[float], p: float) -> float:
    """Centile par rang le plus proche (nearest-rank)."""
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(1, math.ceil(p / 100 * len(xs)))
    return xs[k - 1]


class Bench:
    def __init__(self, args: argparse.Namespace, client: httpx.AsyncClient):
        self.a = args
        self.client = client
        self.chars_per_token = args.chars_per_token
        self.inflight = 0
        self.system_prompt = ""
        self.repos: list[str] = []
        self.repo_tokens: list[int] = []

    # ---- préparation -----------------------------------------------------

    def build_corpus(self) -> None:
        a = self.a
        cpt = self.chars_per_token
        self.system_prompt = (
            "You are a coding assistant working inside an IDE. "
            "Answer precisely and keep the code style of the repository.\n\n"
            "Available tools and conventions:\n"
            + synth_text("system-prompt", int(a.system_tokens * cpt))
        )
        rnd = random.Random(a.seed)
        self.repos, self.repo_tokens = [], []
        for i in range(a.repo_pool):
            n_tok = int(rnd.triangular(a.repo_tokens_min, a.repo_tokens_max,
                                       a.repo_tokens_mode))
            self.repo_tokens.append(n_tok)
            self.repos.append(synth_text(f"repo-{a.seed}-{i}", int(n_tok * cpt)))

    async def calibrate(self) -> None:
        """Mesure le ratio caractères/token du serveur sur le texte synthétique."""
        if self.a.chars_per_token_fixed:
            return
        small = synth_text("calibration-small", 2_000)
        big = synth_text("calibration-big", 22_000)
        try:
            t_small = await self._count_prompt_tokens(small)
            t_big = await self._count_prompt_tokens(big)
        except Exception as e:  # noqa: BLE001
            print(f"# calibration impossible ({e!r}), ratio par défaut "
                  f"{self.chars_per_token}", file=sys.stderr)
            return
        if t_big > t_small > 0:
            # La différence élimine le surcoût fixe du chat template.
            self.chars_per_token = (len(big) - len(small)) / (t_big - t_small)
            print(f"# calibration : {self.chars_per_token:.2f} caractères/token")
        else:
            print("# calibration : usage non renvoyé, ratio par défaut "
                  f"{self.chars_per_token}", file=sys.stderr)

    async def _count_prompt_tokens(self, text: str) -> int:
        r = await self.client.post("/chat/completions", json={
            "model": self.a.model,
            "messages": [{"role": "user", "content": text}],
            "max_tokens": 1, "stream": False,
        }, timeout=self.a.timeout)
        r.raise_for_status()
        return int(r.json().get("usage", {}).get("prompt_tokens", 0))

    def base_messages(self, repo_idx: int) -> list[dict]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Repository context:\n" + self.repos[repo_idx]},
            {"role": "assistant", "content": "Context loaded."},
        ]

    # ---- une requête -----------------------------------------------------

    def draw_output_tokens(self, rnd: random.Random) -> int:
        a = self.a
        n = rnd.lognormvariate(math.log(a.output_median), a.output_sigma)
        return int(min(a.output_max, max(16, n)))

    async def request(self, messages: list[dict], max_tokens: int, *, step_users: int,
                      user: int, session: int, turn: int, t_step0: float
                      ) -> tuple[Sample, str]:
        a = self.a
        body = {
            "model": a.model, "messages": messages, "stream": True,
            "max_tokens": max_tokens, "temperature": a.temperature,
            "stream_options": {"include_usage": True},
        }
        if a.ignore_eos:
            body["ignore_eos"] = True      # extension vLLM : longueur de sortie maîtrisée
        self.inflight += 1
        inflight = self.inflight
        t0 = time.perf_counter()
        ttft = 0.0
        chunks = 0
        usage: dict = {}
        text: list[str] = []
        ok, error = True, ""
        try:
            async with self.client.stream("POST", "/chat/completions", json=body,
                                          timeout=a.timeout) as r:
                if r.status_code != 200:
                    detail = (await r.aread()).decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"HTTP {r.status_code}: {detail}")
                async for line in r.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    chunk = json.loads(data)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices") or []:
                        delta = choice.get("delta") or {}
                        # Le premier token peut être du raisonnement (reasoning parser).
                        piece = (delta.get("content") or delta.get("reasoning_content")
                                 or delta.get("reasoning"))
                        if piece:
                            if not ttft:
                                ttft = time.perf_counter() - t0
                            chunks += 1
                            if delta.get("content"):
                                text.append(delta["content"])
        except Exception as e:  # noqa: BLE001
            ok, error = False, repr(e)[:300]
        finally:
            self.inflight -= 1
        latency = time.perf_counter() - t0
        completion = int(usage.get("completion_tokens") or chunks)
        details = usage.get("prompt_tokens_details") or {}
        cached = details.get("cached_tokens")
        if ok and not ttft:
            ok, error = False, "aucun token reçu"
        tpot = (latency - ttft) / (completion - 1) * 1000 if ok and completion > 1 else 0.0
        sample = Sample(
            step_users=step_users, user=user, session=session, turn=turn,
            started_s=round(t0 - t_step0, 3), inflight=inflight,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            cached_tokens=-1 if cached is None else int(cached),
            completion_tokens=completion, ttft_s=round(ttft, 4),
            latency_s=round(latency, 4), tpot_ms=round(tpot, 3), ok=ok, error=error,
        )
        return sample, "".join(text)

    # ---- un utilisateur ---------------------------------------------------

    async def user_loop(self, user: int, step_users: int, t_step0: float,
                        stop_at: float, sink) -> None:
        a = self.a
        rnd = random.Random(f"{a.seed}-{step_users}-{user}")
        await asyncio.sleep(rnd.uniform(0, a.ramp_time))
        session = 0
        while time.perf_counter() < stop_at:
            session += 1
            repo_idx = rnd.randrange(a.repo_pool)
            messages = self.base_messages(repo_idx)
            est_tokens = a.system_tokens + self.repo_tokens[repo_idx]
            for turn in range(1, a.turns + 1):
                if time.perf_counter() >= stop_at:
                    return
                turn_tokens = rnd.randint(a.turn_tokens_min, a.turn_tokens_max)
                max_tokens = self.draw_output_tokens(rnd)
                if est_tokens + turn_tokens + max_tokens > a.max_context:
                    break   # contexte plein : nouvelle session
                question = (
                    f"Open file for turn {turn}:\n"
                    + synth_text(f"u{user}-s{session}-t{turn}-{step_users}",
                                 int(turn_tokens * self.chars_per_token))
                    + "\n\nExplain what this code does and propose a refactoring."
                )
                messages.append({"role": "user", "content": question})
                sample, answer = await self.request(
                    messages, max_tokens, step_users=step_users, user=user,
                    session=session, turn=turn, t_step0=t_step0)
                sink(sample)
                if not sample.ok:
                    await asyncio.sleep(min(5.0, max(0.0, stop_at - time.perf_counter())))
                    break
                messages.append({"role": "assistant", "content": answer or "ok"})
                est_tokens += turn_tokens + sample.completion_tokens
                think = rnd.uniform(a.think_min, a.think_max)
                remaining = stop_at - time.perf_counter()
                if remaining <= 0:
                    return
                await asyncio.sleep(min(think, remaining))

    # ---- préchauffage ------------------------------------------------------

    async def warmup(self) -> None:
        a = self.a
        if a.no_warmup:
            return
        print(f"# préchauffage du cache : {a.repo_pool} contextes "
              f"(concurrence {a.warmup_concurrency})")
        sem = asyncio.Semaphore(a.warmup_concurrency)
        t0 = time.perf_counter()

        async def one(i: int) -> None:
            async with sem:
                msgs = self.base_messages(i) + [
                    {"role": "user", "content": "Reply with OK."}]
                s, _ = await self.request(msgs, 4, step_users=0, user=-1, session=0,
                                          turn=0, t_step0=t0)
                state = f"{s.ttft_s:.1f}s" if s.ok else f"ERREUR {s.error}"
                print(f"#   contexte {i:>2} : {s.prompt_tokens} tokens, TTFT {state}")

        await asyncio.gather(*(one(i) for i in range(a.repo_pool)))

    # ---- un palier ---------------------------------------------------------

    async def run_step(self, n_users: int, sink) -> tuple[list[Sample], float]:
        a = self.a
        samples: list[Sample] = []

        def collect(s: Sample) -> None:
            samples.append(s)
            sink(s)

        t0 = time.perf_counter()
        stop_at = t0 + a.step_duration
        await asyncio.gather(*(
            self.user_loop(u, n_users, t0, stop_at, collect) for u in range(n_users)))
        return samples, time.perf_counter() - t0


# --------------------------------------------------------------------------
# Synthèse
# --------------------------------------------------------------------------

SUMMARY_FIELDS = [
    "users", "requests", "errors", "error_rate", "req_per_min", "inflight_max",
    "ttft_p50_s", "ttft_p90_s", "ttft_p99_s", "tpot_p50_ms", "tpot_p90_ms",
    "e2e_p50_s", "e2e_p90_s", "prompt_tokens_mean", "completion_tokens_mean",
    "cache_hit_ratio", "prompt_tok_s", "gen_tok_s", "slo_ok", "slo_breaches",
]


def summarize(n_users: int, samples: list[Sample], duration: float,
              a: argparse.Namespace) -> dict:
    ok = [s for s in samples if s.ok]
    ttft = [s.ttft_s for s in ok]
    tpot = [s.tpot_ms for s in ok if s.completion_tokens > 1]
    e2e = [s.latency_s for s in ok]
    p_tok = sum(s.prompt_tokens for s in ok)
    c_tok = sum(s.completion_tokens for s in ok)
    with_cache = [s for s in ok if s.cached_tokens >= 0 and s.prompt_tokens > 0]
    hit = (sum(s.cached_tokens for s in with_cache)
           / sum(s.prompt_tokens for s in with_cache)) if with_cache else None
    n = len(samples)
    row = {
        "users": n_users,
        "requests": n,
        "errors": n - len(ok),
        "error_rate": round((n - len(ok)) / n, 4) if n else 0.0,
        "req_per_min": round(n / duration * 60, 1) if duration else 0.0,
        "inflight_max": max((s.inflight for s in samples), default=0),
        "ttft_p50_s": round(pct(ttft, 50), 2),
        "ttft_p90_s": round(pct(ttft, 90), 2),
        "ttft_p99_s": round(pct(ttft, 99), 2),
        "tpot_p50_ms": round(pct(tpot, 50), 1),
        "tpot_p90_ms": round(pct(tpot, 90), 1),
        "e2e_p50_s": round(pct(e2e, 50), 1),
        "e2e_p90_s": round(pct(e2e, 90), 1),
        "prompt_tokens_mean": round(p_tok / len(ok)) if ok else 0,
        "completion_tokens_mean": round(c_tok / len(ok)) if ok else 0,
        "cache_hit_ratio": "" if hit is None else round(hit, 3),
        "prompt_tok_s": round(p_tok / duration) if duration else 0,
        "gen_tok_s": round(c_tok / duration) if duration else 0,
    }
    breaches = []
    if not ok:
        breaches.append("aucune requête réussie")
    if row["ttft_p90_s"] > a.slo_ttft_p90:
        breaches.append(f"TTFT P90 {row['ttft_p90_s']}s > {a.slo_ttft_p90}s")
    if row["tpot_p50_ms"] > a.slo_tpot_p50:
        breaches.append(f"TPOT P50 {row['tpot_p50_ms']}ms > {a.slo_tpot_p50}ms")
    if row["error_rate"] > a.slo_error_rate:
        breaches.append(f"erreurs {row['error_rate']:.1%} > {a.slo_error_rate:.1%}")
    row["slo_ok"] = not breaches
    row["slo_breaches"] = "; ".join(breaches)
    return row


def print_row(row: dict) -> None:
    hit = row["cache_hit_ratio"]
    hit_txt = "n/a" if hit == "" else f"{hit:.0%}"
    print(
        f"  {row['users']:>4} users | {row['requests']:>5} req ({row['req_per_min']}/min, "
        f"{row['errors']} err) | en vol max {row['inflight_max']} | "
        f"TTFT p50 {row['ttft_p50_s']}s p90 {row['ttft_p90_s']}s p99 {row['ttft_p99_s']}s | "
        f"TPOT p50 {row['tpot_p50_ms']}ms p90 {row['tpot_p90_ms']}ms | "
        f"e2e p90 {row['e2e_p90_s']}s | cache {hit_txt} | "
        f"{row['prompt_tok_s']} tok/s in, {row['gen_tok_s']} tok/s out | "
        + ("SLO OK" if row["slo_ok"] else "SLO KO : " + row["slo_breaches"])
    )


# --------------------------------------------------------------------------
# Programme principal
# --------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark de montée en charge par paliers (API OpenAI-compatible).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("cible")
    g.add_argument("--url", default="http://127.0.0.1:8020/v1", help="base URL de l'API")
    g.add_argument("--model", default="qwen3.8-27b-fp8")
    g.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""),
                   help="sinon variable OPENAI_API_KEY")
    g.add_argument("--ca-bundle", default="", help="fichier CA pour un certificat interne")
    g.add_argument("--timeout", type=float, default=900.0, help="timeout par requête (s)")

    g = p.add_argument_group("charge")
    g.add_argument("--steps", default="10,20,40,60,80,100,150",
                   help="utilisateurs simultanés par palier")
    g.add_argument("--step-duration", type=float, default=600.0, help="durée d'un palier (s)")
    g.add_argument("--ramp-time", type=float, default=30.0,
                   help="étalement des démarrages au début de chaque palier (s)")
    g.add_argument("--think-min", type=float, default=20.0, help="réflexion entre tours (s)")
    g.add_argument("--think-max", type=float, default=60.0)
    g.add_argument("--turns", type=int, default=5, help="tours par session")
    g.add_argument("--stop-on-breach", action="store_true",
                   help="arrêter au premier palier hors SLO")

    g = p.add_argument_group("profil des prompts")
    g.add_argument("--repo-pool", type=int, default=12,
                   help="nombre de contextes de repo distincts partagés entre utilisateurs")
    g.add_argument("--repo-tokens-min", type=int, default=60_000)
    g.add_argument("--repo-tokens-mode", type=int, default=95_000)
    g.add_argument("--repo-tokens-max", type=int, default=180_000)
    g.add_argument("--system-tokens", type=int, default=3_000)
    g.add_argument("--turn-tokens-min", type=int, default=1_000)
    g.add_argument("--turn-tokens-max", type=int, default=6_000)
    g.add_argument("--max-context", type=int, default=250_000,
                   help="au-delà, l'utilisateur ouvre une nouvelle session")
    g.add_argument("--output-median", type=float, default=200.0, help="tokens de sortie (médiane)")
    g.add_argument("--output-sigma", type=float, default=1.48,
                   help="écart-type log-normal (1.48 -> moyenne ~600, P90 ~1300)")
    g.add_argument("--output-max", type=int, default=4_000)
    g.add_argument("--temperature", type=float, default=0.6)
    g.add_argument("--no-ignore-eos", dest="ignore_eos", action="store_false",
                   help="ne pas envoyer ignore_eos (longueur de sortie alors décidée par le modèle)")
    g.add_argument("--chars-per-token", type=float, default=3.0,
                   help="ratio utilisé si la calibration échoue")
    g.add_argument("--chars-per-token-fixed", action="store_true",
                   help="ne pas calibrer, utiliser --chars-per-token tel quel")
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("préchauffage")
    g.add_argument("--no-warmup", action="store_true")
    g.add_argument("--warmup-concurrency", type=int, default=2)

    g = p.add_argument_group("seuils (SLO)")
    g.add_argument("--slo-ttft-p90", type=float, default=10.0, help="secondes")
    g.add_argument("--slo-tpot-p50", type=float, default=20.0, help="millisecondes")
    g.add_argument("--slo-error-rate", type=float, default=0.01)

    g = p.add_argument_group("sortie")
    g.add_argument("--out", default="", help="dossier de résultats (défaut : results/<date>)")
    g.add_argument("--dry-run", action="store_true",
                   help="afficher le profil généré sans envoyer de requête")
    a = p.parse_args(argv)
    try:
        a.step_list = [int(x) for x in a.steps.split(",") if x.strip()]
    except ValueError:
        p.error("--steps attend des entiers séparés par des virgules")
    if not a.step_list or min(a.step_list) < 1:
        p.error("--steps doit contenir au moins un entier positif")
    if not (a.repo_tokens_min <= a.repo_tokens_mode <= a.repo_tokens_max):
        p.error("il faut repo-tokens-min <= repo-tokens-mode <= repo-tokens-max")
    if a.think_min > a.think_max or a.turn_tokens_min > a.turn_tokens_max:
        p.error("un minimum est supérieur à son maximum")
    return a


async def amain(a: argparse.Namespace) -> int:
    headers = {"Authorization": f"Bearer {a.api_key}"} if a.api_key else {}
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
    verify: bool | str = a.ca_bundle or True
    async with httpx.AsyncClient(base_url=a.url.rstrip("/"), headers=headers,
                                 limits=limits, verify=verify) as client:
        bench = Bench(a, client)
        if not a.dry_run:
            await bench.calibrate()
        bench.build_corpus()
        toks = sorted(bench.repo_tokens)
        print(f"# profil : {a.repo_pool} contextes, {toks[0]}–{toks[-1]} tokens "
              f"(moyenne {sum(toks) // len(toks)}), system {a.system_tokens}, "
              f"{a.turns} tours/session, sortie médiane {a.output_median:.0f} tokens")
        print(f"# paliers : {a.step_list} utilisateurs, {a.step_duration:.0f}s chacun, "
              f"réflexion {a.think_min:g}–{a.think_max:g}s")
        if a.dry_run:
            return 0

        out_dir = Path(a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "config.json").write_text(
            json.dumps({k: v for k, v in vars(a).items() if k != "api_key"}, indent=2))
        rows: list[dict] = []
        with open(out_dir / "requests.jsonl", "w") as req_file:
            def sink(s: Sample) -> None:
                req_file.write(json.dumps(asdict(s)) + "\n")
                req_file.flush()

            await bench.warmup()
            try:
                for n_users in a.step_list:
                    print(f"# palier {n_users} utilisateurs…")
                    samples, duration = await bench.run_step(n_users, sink)
                    row = summarize(n_users, samples, duration, a)
                    rows.append(row)
                    print_row(row)
                    if a.stop_on_breach and not row["slo_ok"]:
                        print("# arrêt : palier hors SLO")
                        break
            except (KeyboardInterrupt, asyncio.CancelledError):
                print("# interrompu, écriture des résultats partiels")
            finally:
                with open(out_dir / "summary.csv", "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
                    w.writeheader()
                    w.writerows(rows)

        good = [r["users"] for r in rows if r["slo_ok"]]
        bad = [r["users"] for r in rows if not r["slo_ok"]]
        print(f"# résultats : {out_dir}/summary.csv et requests.jsonl")
        if rows:
            first_bad = min(bad) if bad else None
            held = [u for u in good if first_bad is None or u < first_bad]
            if held:
                print(f"# capacité : SLO tenus jusqu'à {max(held)} utilisateurs simultanés"
                      + (f", rupture à {first_bad}" if first_bad else
                         " (aucune rupture : ajoute des paliers plus hauts)"))
            else:
                print(f"# capacité : SLO non tenus dès {first_bad} utilisateurs")
    return 0


def main() -> None:
    args = parse_args()
    try:
        sys.exit(asyncio.run(amain(args)))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
