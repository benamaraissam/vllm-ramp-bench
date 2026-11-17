# vllm-ramp-bench

Benchmark de montée en charge pour une API OpenAI-compatible (vLLM, Ray Serve LLM).
Il sert à répondre à une question : **combien d'utilisateurs simultanés la
configuration tient-elle avant que la latence décroche ?**

Le trafic simulé est celui d'un assistant de code :

- prompts longs : un contexte de repo de 60k à 180k tokens (moyenne ~100k) ;
- sessions de plusieurs tours sur le même contexte, donc un prefix cache
  sollicité comme en production ;
- sorties courtes : médiane ~200 tokens, moyenne ~600, P90 ~1300 ;
- utilisateurs en boucle fermée : chacun attend sa réponse, réfléchit, puis
  envoie le tour suivant.

## Installation

```bash
pip install -r requirements.txt   # httpx uniquement, Python 3.10+
```

## Utilisation

```bash
# 1. Vérifier le profil généré, sans rien envoyer
python bench.py --dry-run

# 2. Essai à blanc contre le faux serveur (aucun GPU nécessaire)
python mock_server.py --port 8021 &
python bench.py --url http://127.0.0.1:8021/v1 --steps 2,4 --step-duration 20 \
    --think-min 1 --think-max 2

# 3. Vrai test
export OPENAI_API_KEY=...          # si l'API est protégée
python bench.py --url http://127.0.0.1:8020/v1 --model qwen3.8-27b-fp8 \
    --steps 10,20,40,60,80,100,150 --step-duration 600 --stop-on-breach
```

> **Attention :** ce test envoie des millions de tokens. Lancé contre la
> production, il dégrade le service des vrais utilisateurs. À faire sur un
> environnement de test ou en dehors des heures d'usage.

## Déroulement

1. **Calibration** : deux petites requêtes mesurent le ratio caractères/token
   du serveur, pour que les tailles demandées en tokens soient respectées.
2. **Préchauffage** : chaque contexte de repo est envoyé une fois, pour partir
   d'un cache chaud comme en production (`--no-warmup` pour mesurer à froid).
3. **Paliers** : pour chaque valeur de `--steps`, N utilisateurs tournent
   pendant `--step-duration` secondes. Les démarrages sont étalés sur
   `--ramp-time` secondes.
4. **Synthèse** : une ligne par palier, avec le verdict sur les seuils.

## Résultats

Dans `results/<date>/` :

| Fichier | Contenu |
|---|---|
| `summary.csv` | une ligne par palier |
| `requests.jsonl` | une ligne par requête (tokens, TTFT, latence, erreur) |
| `config.json` | les paramètres du test (sans la clé d'API) |

Colonnes principales de `summary.csv` :

| Colonne | Signification |
|---|---|
| `users` | utilisateurs simultanés du palier |
| `req_per_min`, `inflight_max` | débit servi, maximum de requêtes en vol |
| `ttft_p50_s`, `ttft_p90_s`, `ttft_p99_s` | délai avant le premier token |
| `tpot_p50_ms`, `tpot_p90_ms` | temps par token généré |
| `e2e_p50_s`, `e2e_p90_s` | latence totale d'une requête |
| `cache_hit_ratio` | part des tokens du prompt servie par le cache, si le serveur la renvoie |
| `prompt_tok_s`, `gen_tok_s` | tokens d'entrée et de sortie par seconde |
| `slo_ok`, `slo_breaches` | seuils tenus ou non, et lesquels cassent |

La capacité est le dernier palier où `slo_ok` vaut `True` avant la première
rupture. Seuils par défaut : TTFT P90 ≤ 10 s, TPOT P50 ≤ 20 ms, erreurs ≤ 1 %
(`--slo-ttft-p90`, `--slo-tpot-p50`, `--slo-error-rate`).

## Régler le profil

| Option | Défaut | Rôle |
|---|---|---|
| `--repo-pool` | 12 | contextes distincts partagés entre utilisateurs |
| `--repo-tokens-min/-mode/-max` | 60k / 95k / 180k | taille des contextes |
| `--turn-tokens-min/-max` | 1k / 6k | tokens ajoutés à chaque tour |
| `--turns` | 5 | tours par session |
| `--think-min/-max` | 20 / 60 s | réflexion entre deux tours |
| `--output-median`, `--output-sigma` | 200, 1.48 | longueur des sorties (log-normale) |

`python bench.py --help` liste toutes les options.

Deux réglages méritent une vérification sur votre plateforme :

- **`--repo-pool`** fixe le taux de hit du cache. Comparez `cache_hit_ratio`
  (ou le panneau « GPU KV Cache Hit Rate » de Grafana) à celui de la
  production : s'il est nettement plus bas, réduisez le pool ; s'il est plus
  haut, augmentez-le.
- **`ignore_eos`** est envoyé par défaut pour que la sortie fasse exactement
  la longueur tirée. C'est une extension vLLM : avec un autre serveur,
  ajoutez `--no-ignore-eos`.

## Comparer deux configurations

Lancez le même test avant et après le changement (par exemple
`max_num_batched_tokens`, ou TP2 contre TP4), avec la même `--seed`, et
comparez les `summary.csv`. Ne changez qu'un paramètre serveur à la fois.

## Limites

- Le texte est du pseudo-code synthétique : les tailles et la structure des
  prompts sont réalistes, pas leur contenu.
- Le client mesure la latence vue de l'extérieur, réseau et proxy compris.
  Gardez Grafana ouvert pour les métriques internes (file d'attente,
  préemptions, utilisation du KV cache).
- Pas de requêtes multimodales ni d'appels d'outils.
