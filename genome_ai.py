#!/usr/bin/env python3
"""
genome_ai.py -- local-LLM question answering for the codon frequency analyzer.

Talks to an Ollama server running on this machine, the same way Open WebUI does:
plain HTTP to the Ollama REST API (default http://localhost:11434), honouring the
OLLAMA_HOST environment variable. A unix socket is supported too
(OLLAMA_HOST=unix:///var/run/ollama.sock). No API key, no cloud, no SDK --
standard library only.

Retrieval (RAG) runs before the prompt and uses only key-free public endpoints:
  1. NCBI nuccore  -- record metadata for the accession in the FASTA header
  2. NCBI taxonomy -- lineage for that organism
  3. PubMed        -- recent literature on the organism
  4. Wikipedia     -- summary article, when one exists
  5. DuckDuckGo    -- general web results (best effort; HTML scrape)

Set your question in USER_PROMPT below, or pass --ask on the command line.
"""

import http.client
import json
import os
import re
import socket
import sys
import urllib.parse
import urllib.request

# ---------------------------------------------------------------------------
# EDIT ME: the question the AI will answer about the genome.
# ---------------------------------------------------------------------------
USER_PROMPT = "What virus is this, and what does its codon usage bias suggest about its host and its genome composition?"
# ---------------------------------------------------------------------------

# Ollama connection. OLLAMA_HOST matches what the ollama CLI and Open WebUI use.
#   http://localhost:11434          (default)
#   http://192.168.1.50:11434       (remote box)
#   unix:///var/run/ollama.sock     (socket-activated service)
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:26b")

NCBI = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
UA = "codon-freq-analyzer/1.0 (+local research tool)"
HTTP_TIMEOUT = 25

SYSTEM_PROMPT = """\
You are a computational virology assistant embedded in a codon-usage analysis tool.

You are given codon counts computed from a genome FASTA file, basic sequence
statistics, and retrieved reference material from NCBI, PubMed and the web.

How to answer:
- Ground every factual claim in the supplied statistics or the retrieved
  material. When you use a retrieved source, name it.
- Separate what the numbers in THIS run show from background knowledge about
  the organism.
- If the retrieved material does not answer the question, say so. Do not invent
  accession numbers, citations, host species, or figures. "The data here does
  not show that" is a correct and complete answer when it is true.
- Method caveat worth repeating when relevant: a sliding window with step 1
  counts nucleotide triplets in all three forward frames and ignores the reverse
  strand, so these are 3-mer frequencies, not the translated codon usage of
  annotated coding sequences. Step 3 is frame 1 only and is still not CDS-aware.

Scope: genome composition, codon and nucleotide usage, taxonomy, molecular
virology, and the analysis method itself.

You do not give medical, clinical, or dietary advice, and you do not accept the
premise that codon or amino acid frequencies indicate what a patient should eat
or how to recover from an infection. No such link exists: viral proteins are
translated from the host's own amino acid pool by the host's ribosomes, that
pool cannot be selectively depleted by diet, and those same amino acids build
the immune response. If a question rests on that premise, say so plainly,
explain briefly why it fails, answer whatever genomics question remains, and
refer clinical questions to a clinician.\
"""


# ===========================================================================
# Ollama transport -- TCP or unix socket, like Open WebUI's Ollama connector
# ===========================================================================

class _UnixHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection over an AF_UNIX socket."""

    def __init__(self, socket_path, timeout=HTTP_TIMEOUT):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.socket_path)
        self.sock = sock


def _connect(host=None, timeout=HTTP_TIMEOUT):
    """Return (connection, host_header) for the configured Ollama endpoint."""
    host = host or OLLAMA_HOST
    if host.startswith("unix://"):
        return _UnixHTTPConnection(host[len("unix://"):], timeout), "localhost"
    if host.startswith("/") and os.path.exists(host):          # bare socket path
        return _UnixHTTPConnection(host, timeout), "localhost"
    if "://" not in host:                                       # "localhost:11434"
        host = "http://" + host
    u = urllib.parse.urlparse(host)
    port = u.port or (443 if u.scheme == "https" else 11434)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, port, timeout=timeout), u.netloc


def _ollama_request(method, path, body=None, host=None, timeout=HTTP_TIMEOUT):
    conn, host_header = _connect(host, timeout)
    headers = {"Host": host_header, "Accept": "application/json",
               "User-Agent": UA}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    conn.request(method, path, body=payload, headers=headers)
    return conn, conn.getresponse()


def ollama_available(host=None):
    """Return (ok, detail). Used to give a clear message when nothing is listening."""
    try:
        conn, resp = _ollama_request("GET", "/api/tags", host=host, timeout=5)
        try:
            if resp.status != 200:
                return False, f"Ollama returned HTTP {resp.status}"
            names = [m["name"] for m in json.load(resp).get("models", [])]
        finally:
            conn.close()
        return True, names
    except (OSError, ValueError) as e:
        return False, f"{type(e).__name__}: {e}"


def ollama_chat(messages, model=None, host=None, temperature=0.2,
                num_ctx=16384, stream=True, out=sys.stdout):
    """POST /api/chat and return the assistant's full reply.

    Streams NDJSON exactly like Open WebUI: one JSON object per line, the text
    arriving in message.content, terminated by an object with done=true.
    """
    model = model or OLLAMA_MODEL
    body = {
        "model": model,
        "messages": messages,
        "stream": bool(stream),
        "options": {"temperature": temperature, "num_ctx": num_ctx},
    }

    try:
        conn, resp = _ollama_request("POST", "/api/chat", body, host,
                                     timeout=600)
    except OSError as e:
        _die_no_server(host, f"{type(e).__name__}: {e}")

    try:
        if resp.status == 404:
            detail = resp.read().decode(errors="replace")
            ok, names = ollama_available(host)
            hint = ""
            if ok and isinstance(names, list):
                hint = ("\nModels this server has:\n  " + "\n  ".join(names)
                        if names else "\nThis server has no models pulled yet.")
            sys.exit(f"Ollama has no model named {model!r}.\n{detail.strip()}"
                     f"{hint}\n\nPull it with:  ollama pull {model}\n"
                     f"Or choose another:  --model <name>  (or OLLAMA_MODEL=<name>)")
        if resp.status != 200:
            sys.exit(f"Ollama error HTTP {resp.status}: "
                     f"{resp.read().decode(errors='replace')[:500]}")

        if not stream:
            data = json.load(resp)
            text = data.get("message", {}).get("content", "")
            out.write(text + "\n")
            return text

        parts = []
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in chunk:
                sys.exit(f"Ollama error: {chunk['error']}")
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                parts.append(piece)
                out.write(piece)
                out.flush()
            if chunk.get("done"):
                break
        out.write("\n")
        return "".join(parts)
    finally:
        conn.close()


def _die_no_server(host, detail):
    sys.exit(
        f"Could not reach Ollama at {host or OLLAMA_HOST} ({detail}).\n"
        "  - Is it running?   ollama serve\n"
        "  - Check the port:  curl http://localhost:11434/api/tags\n"
        "  - Different host?  export OLLAMA_HOST=http://host:11434\n"
        "  - Unix socket?     export OLLAMA_HOST=unix:///var/run/ollama.sock"
    )


# ===========================================================================
# Retrieval -- key-free public endpoints
# ===========================================================================

def _get_json(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _get_text(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_accession(header):
    """Pull a GenBank/RefSeq accession out of a FASTA header line."""
    m = re.match(r"^\s*([A-Z]{1,2}_?[A-Z]{0,3}\d{5,8}(?:\.\d+)?)\b", header)
    return m.group(1) if m else None


def ncbi_nuccore(accession):
    url = (f"{NCBI}/esummary.fcgi?db=nuccore"
           f"&id={urllib.parse.quote(accession)}&retmode=json")
    payload = _get_json(url)
    result = payload.get("result", {})
    uids = result.get("uids") or []
    if not uids:
        return None
    rec = result[uids[0]]
    return {k: rec[k] for k in
            ("accessionversion", "title", "organism", "taxid", "slen",
             "biomol", "moltype", "genome", "subname", "updatedate")
            if rec.get(k)}


def ncbi_taxonomy(taxid):
    url = f"{NCBI}/esummary.fcgi?db=taxonomy&id={taxid}&retmode=json"
    rec = _get_json(url)["result"][str(taxid)]
    return {k: rec[k] for k in ("scientificname", "rank", "division",
                                "genbankdivision", "lineage") if rec.get(k)}


def pubmed(term, retmax=5):
    ids = _get_json(f"{NCBI}/esearch.fcgi?db=pubmed&retmode=json&retmax={retmax}"
                    f"&term={urllib.parse.quote(term)}"
                    )["esearchresult"].get("idlist", [])
    if not ids:
        return []
    res = _get_json(f"{NCBI}/esummary.fcgi?db=pubmed&retmode=json"
                    f"&id={','.join(ids)}")["result"]
    out = []
    for uid in res.get("uids", []):
        r = res[uid]
        authors = ", ".join(a["name"] for a in r.get("authors", [])[:3])
        out.append({"pmid": uid, "title": r.get("title", ""),
                    "journal": r.get("source", ""), "date": r.get("pubdate", ""),
                    "authors": authors,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{uid}/"})
    return out


def wikipedia(title):
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + urllib.parse.quote(title.replace(" ", "_")))
    d = _get_json(url)
    if d.get("type", "").endswith("not_found"):
        return None
    return {"title": d.get("title"), "extract": d.get("extract"),
            "url": d.get("content_urls", {}).get("desktop", {}).get("page")}


def duckduckgo(query, n=5):
    """Best-effort general web search. HTML scrape -- no key, but fragile."""
    html = _get_text("https://html.duckduckgo.com/html/?q="
                     + urllib.parse.quote(query))
    hits, seen = [], set()
    for href, title in re.findall(
            r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
            html, re.S):
        url = urllib.parse.unquote(
            urllib.parse.parse_qs(urllib.parse.urlparse(href).query)
            .get("uddg", [href])[0])
        text = re.sub(r"<[^>]+>", "", title).strip()
        if url not in seen:
            seen.add(url)
            hits.append({"title": text, "url": url})
        if len(hits) >= n:
            break
    return hits


def retrieve(records, extra_query=None, verbose=True):
    """Run every retrieval source. Each failure degrades to a note, never a crash."""
    def log(msg):
        if verbose:
            print(f"  {msg}", file=sys.stderr)

    lines, organism, sources = ["=== RETRIEVED REFERENCE MATERIAL ==="], None, []
    accessions = []
    for header, _ in records:
        acc = parse_accession(header)
        if acc and acc not in accessions:
            accessions.append(acc)

    # 1 + 2: NCBI record and taxonomy
    for acc in accessions:
        lines.append(f"\n[NCBI nuccore {acc}]")
        try:
            rec = ncbi_nuccore(acc)
        except Exception as e:
            lines.append(f"  lookup failed: {type(e).__name__}: {e}")
            log(f"NCBI nuccore {acc}: failed")
            continue
        if not rec:
            lines.append("  no record found")
            continue
        for k, v in rec.items():
            lines.append(f"  {k}: {v}")
        sources.append(f"https://www.ncbi.nlm.nih.gov/nuccore/{acc}")
        log(f"NCBI nuccore {acc}: ok")
        organism = organism or rec.get("organism")
        if rec.get("taxid"):
            try:
                tax = ncbi_taxonomy(rec["taxid"])
                lines.append(f"\n[NCBI taxonomy {rec['taxid']}]")
                for k, v in tax.items():
                    lines.append(f"  {k}: {v}")
                sources.append(
                    f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={rec['taxid']}")
                log(f"NCBI taxonomy {rec['taxid']}: ok")
            except Exception as e:
                lines.append(f"  taxonomy lookup failed: {type(e).__name__}: {e}")

    if not accessions:
        lines.append("\n  (no accession could be parsed from the FASTA headers)")

    topic = organism or (records[0][0][:80] if records else "")
    query = extra_query or topic

    # 3: PubMed
    if topic:
        lines.append(f"\n[PubMed: {topic}]")
        try:
            papers = pubmed(f"{topic}[All Fields]")
            if papers:
                for p in papers:
                    lines.append(f"  - {p['title']}")
                    lines.append(f"    {p['authors']} | {p['journal']} {p['date']} | {p['url']}")
                    sources.append(p["url"])
                log(f"PubMed: {len(papers)} result(s)")
            else:
                lines.append("  no results")
                log("PubMed: no results")
        except Exception as e:
            lines.append(f"  search failed: {type(e).__name__}: {e}")
            log("PubMed: failed")

    # 4: Wikipedia
    if topic:
        lines.append(f"\n[Wikipedia: {topic}]")
        try:
            w = wikipedia(topic)
            if w and w.get("extract"):
                lines.append(f"  {w['title']}: {w['extract']}")
                lines.append(f"  {w['url']}")
                sources.append(w["url"])
                log("Wikipedia: ok")
            else:
                lines.append("  no article")
                log("Wikipedia: no article")
        except Exception as e:
            lines.append(f"  lookup failed: {type(e).__name__}: {e}")
            log("Wikipedia: failed")

    # 5: general web
    if query:
        lines.append(f"\n[Web search: {query}]")
        try:
            hits = duckduckgo(f"{query} genome codon usage")
            if hits:
                for h in hits:
                    lines.append(f"  - {h['title']}\n    {h['url']}")
                    sources.append(h["url"])
                log(f"Web search: {len(hits)} result(s)")
            else:
                lines.append("  no results")
                log("Web search: no results")
        except Exception as e:
            lines.append(f"  search failed: {type(e).__name__}: {e}")
            log("Web search: failed")

    return "\n".join(lines), sources


# ===========================================================================
# Context building
# ===========================================================================

def build_context(records, counts, skipped, step, top_n=10):
    """Render the analyzer's output as a text block for the model."""
    total = sum(counts.values())
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))

    lines = ["=== SEQUENCE RECORDS ==="]
    for header, seq in records:
        gc = sum(seq.count(b) for b in "GC")
        acgt = sum(seq.count(b) for b in "ACGT")
        gc_pct = f"{100 * gc / acgt:.2f}%" if acgt else "n/a"
        lines.append(f"  {header}")
        lines.append(f"    length={len(seq):,} nt   GC={gc_pct}   "
                     f"ambiguous={len(seq) - acgt:,} nt")

    lines += ["", "=== METHOD ===",
              f"  Sliding window of 3 nt, step {step} "
              f"({'all 3 forward frames' if step == 1 else 'frame 1 only'}, "
              f"forward strand only, not CDS-aware)",
              f"  Windows counted: {total:,}   Skipped (ambiguous base): {skipped:,}",
              "", f"=== TOP {top_n} TRIPLETS ==="]
    for rank, (codon, n) in enumerate(ranked[:top_n], 1):
        pct = 100 * n / total if total else 0.0
        lines.append(f"  {rank:>2}. {codon}  {n:>10,}  {pct:6.3f}%")

    lines += ["", "=== ALL 64 COUNTS (codon:count) ==="]
    lines.append("  " + "  ".join(f"{c}:{n}" for c, n in sorted(counts.items())))
    return "\n".join(lines)


# ===========================================================================
# Entry point
# ===========================================================================

def run(records, counts, skipped, step, question, top_n=10,
        model=None, host=None, no_retrieval=False, temperature=0.2):
    """Called by codon_freq.py --ask."""
    host = host or OLLAMA_HOST
    model = model or OLLAMA_MODEL

    ok, info = ollama_available(host)
    if not ok:
        _die_no_server(host, info)
    if isinstance(info, list) and info:
        base = {n.split(":")[0] for n in info}
        if model not in info and model.split(":")[0] not in base:
            print(f"Note: {model!r} is not in this server's model list "
                  f"({', '.join(info)}). Trying anyway.", file=sys.stderr)

    context = build_context(records, counts, skipped, step, top_n)

    if no_retrieval:
        retrieved, sources = "=== RETRIEVED REFERENCE MATERIAL ===\n  (skipped)", []
    else:
        print("Retrieving reference material ...", file=sys.stderr)
        retrieved, sources = retrieve(records)

    user_block = (f"{context}\n\n{retrieved}\n\n"
                  f"=== QUESTION ===\n{question}")

    print(f"Asking {model} at {host} ...\n", file=sys.stderr)
    print("=" * 70)
    print(f"QUESTION: {question}")
    print("=" * 70)

    ollama_chat([{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": user_block}],
                model=model, host=host, temperature=temperature)

    if sources:
        print("Retrieved sources:")
        for s in dict.fromkeys(sources):
            print(f"  - {s}")


if __name__ == "__main__":
    print(__doc__)
    ok, info = ollama_available()
    print(f"Ollama at {OLLAMA_HOST}: "
          + (f"reachable, models: {', '.join(info) if info else '(none pulled)'}"
             if ok else f"NOT reachable -- {info}"))
    print(f"\nCurrent USER_PROMPT:\n  {USER_PROMPT}")
    print('\nRun via:  python3 codon_freq.py <genome.fna.gz> --ask "your question"')
