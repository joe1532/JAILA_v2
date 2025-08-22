#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal markdown chunker med parent‑noder (stk_parent, paragraf_parent, section_parent).

Kørsel:
  python chunker.py --input "Kildeskatteloven (2025-04-11 nr. 460)_with_markdown.md" \
                    --out-prefix ksl

Output:
  ksl_chunks.jsonl
  ksl_chunks.json

Afhænger kun af standardbiblioteket. Hvis pandas er installeret, bruges det til CSV; ellers falder scriptet tilbage til csv‑modulet.
"""

from __future__ import annotations
import re, os, json, argparse, csv, uuid
from typing import List, Dict, Any, Optional, Tuple
import sys

# LLM imports (optional)
try:
    from fireworks import LLM
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False

# ------------------------------
# Regex‑konfiguration (kan tilpasses)
# ------------------------------
RE_SECTION      = re.compile(r'^#\s*(AFSNIT|KAPITEL|DEL|BILAG)\s+(.+)$', re.IGNORECASE)
RE_PARAGRAF     = re.compile(r'^##\s*§\s*([0-9]+(?:\s*[A-Za-z]+)?)\.\s*(.*)$', re.IGNORECASE)
RE_STK          = re.compile(r'^###\s*stk\.\s*([0-9A-Za-z]+)\.', re.IGNORECASE)
RE_NR           = re.compile(r'^###\s*(\d+(?:\s*[a-zA-Z]+)?)[\)\.]', re.IGNORECASE)
RE_LITRA        = re.compile(r'^####\s*([a-z])[\)\.]', re.IGNORECASE)
RE_PUNKT        = re.compile(r'^#####\s*(\d+)[\)\.]', re.IGNORECASE)
RE_NOTE_BODY    = re.compile(r'^\s*\(\u200B?(\d+)\u200B?\)\s')
RE_MD_PREFIX    = re.compile(r'^(#+)\s*')
RE_INLINE_NOTE  = re.compile(r'\(\u200B?(\d+)\u200B?\)')

ANCHOR_FMT = "⟦{id}⟧"

def generate_chunk_uuid() -> str:
    """Generate full random UUID for chunks to avoid collisions."""
    return str(uuid.uuid4())


def estimate_tokens(text: str) -> int:
    """Forbedret token-estimering med ordtælling som fallback for kortordet tekst."""
    if not text:
        return 0
    char_based = len(text) // 4
    word_count = len(text.split())
    word_based = int(word_count * 1.3)
    return max(char_based, word_based)


def should_split_chunk(text: str, max_tokens: Optional[int]) -> bool:
    """Afgør om en chunk skal splittes baseret på token-begrænsning."""
    if max_tokens is None or max_tokens <= 0:
        return False
    return estimate_tokens(text) > max_tokens


def find_split_points(text: str, max_tokens: int) -> List[int]:
    """Find gode split-punkter i tekst baseret på sætninger og afsnit."""
    if estimate_tokens(text) <= max_tokens:
        return []
    
    split_points = []
    
    paragraphs = text.split('\n\n')
    current_pos = 0
    current_tokens = 0
    
    for i, para in enumerate(paragraphs):
        para_tokens = estimate_tokens(para)
        
        if current_tokens + para_tokens > max_tokens and current_tokens > 0:
            split_points.append(current_pos)
            current_tokens = para_tokens
        else:
            current_tokens += para_tokens
        
        current_pos += len(para) + (2 if i < len(paragraphs) - 1 else 0)
    
    if not split_points and estimate_tokens(text) > max_tokens:
        sentences = text.split('. ')
        current_pos = 0
        current_tokens = 0
        
        for i, sent in enumerate(sentences):
            sent_tokens = estimate_tokens(sent)
            
            if current_tokens + sent_tokens > max_tokens and current_tokens > 0:
                split_points.append(current_pos)
                current_tokens = sent_tokens
            else:
                current_tokens += sent_tokens
            
            current_pos += len(sent) + (2 if i < len(sentences) - 1 else 0)
    
    return split_points


def redistribute_metadata(original_chunk: Dict[str, Any], text_parts: List[str]) -> List[Dict[str, Any]]:
    """Fordel metadata (note_anchors, dom_refs, jv_refs) mellem split-dele."""
    parts = []
    current_pos = 0
    
    for i, text_part in enumerate(text_parts):
        part_start = current_pos
        part_end = current_pos + len(text_part)
        
        part_chunk = original_chunk.copy()
        part_chunk["uuid"] = generate_chunk_uuid()
        part_chunk["text_plain"] = text_part
        part_chunk["part_index"] = i + 1
        part_chunk["part_total"] = len(text_parts)
        part_chunk["split_reason"] = "token_limit"
        
        original_id = part_chunk.get("chunk_id", "")
        part_chunk["chunk_id"] = f"{original_id} (part {i+1})"
        
        original_atom_id = part_chunk.get("atom_id", "")
        if original_atom_id:
            if "--kind" in original_atom_id:
                base_id, kind_part = original_atom_id.rsplit("--kind", 1)
                part_chunk["atom_id"] = f"{base_id}--part{i+1}-of-{len(text_parts)}--kind{kind_part}"
            else:
                part_chunk["atom_id"] = f"{original_atom_id}--part{i+1}-of-{len(text_parts)}"
        
        def filter_metadata_by_position(metadata_list: List[Dict], start: int, end: int) -> List[Dict]:
            filtered = []
            for item in metadata_list:
                item_start = item.get("start", 0)
                if start <= item_start < end:
                    new_item = item.copy()
                    new_item["start"] = item_start - start
                    filtered.append(new_item)
            return filtered
        
        part_chunk["note_anchors"] = filter_metadata_by_position(
            original_chunk.get("note_anchors", []), part_start, part_end
        )
        part_chunk["dom_refs"] = filter_metadata_by_position(
            original_chunk.get("dom_refs", []), part_start, part_end
        )
        part_chunk["jv_refs"] = filter_metadata_by_position(
            original_chunk.get("jv_refs", []), part_start, part_end
        )
        
        parts.append(part_chunk)
        current_pos = part_end
    
    return parts


def split_chunk_by_tokens(chunk: Dict[str, Any], max_tokens: int) -> List[Dict[str, Any]]:
    """Split en chunk i mindre dele baseret på token-begrænsning."""
    text = chunk.get("text_plain", "")
    
    if not should_split_chunk(text, max_tokens):
        return [chunk]
    
    split_points = find_split_points(text, max_tokens)
    
    if not split_points:
        return [chunk]
    
    print(f"🔄 Splitting chunk {chunk.get('chunk_id')} ({estimate_tokens(text)} tokens) into parts...")
    sys.stdout.flush()
    
    text_parts = []
    last_pos = 0
    
    for split_pos in split_points:
        text_parts.append(text[last_pos:split_pos].strip())
        last_pos = split_pos
    
    text_parts.append(text[last_pos:].strip())
    
    text_parts = [part for part in text_parts if part.strip()]
    
    if len(text_parts) <= 1:
        return [chunk]
    
    split_chunks = redistribute_metadata(chunk, text_parts)
    print(f"✅ Split into {len(split_chunks)} parts")
    sys.stdout.flush()
    return split_chunks


def get_fireworks_llm(api_key=None):
    """Initialize Fireworks LLM if available."""
    if not FIREWORKS_AVAILABLE:
        return None
    try:
        if api_key:
            import fireworks
            fireworks.client.api_key = api_key
            import os
            os.environ['FIREWORKS_API_KEY'] = api_key
        
        return LLM(
            model="qwen3-235b-a22b",
            deployment_type="serverless"
        )
    except Exception as e:
        print(f"Warning: Could not initialize Fireworks LLM: {e}")
        return None

def generate_llm_bullets(children_chunks: List[Dict[str, Any]], 
                        parent_type: str = "stk_parent",
                        llm=None) -> str:
    """Generate parent chunk bullets using LLM."""
    if llm is None:
        return None
    
    child_texts = [f"{i}. {child.get('text_plain', '')[:200]}" for i, child in enumerate(children_chunks, 1)]
    children_text = "\n".join(child_texts)
    
    prompt = f"""Du er en ekspert i dansk skatteret. Lav en præcis oversigt af følgende juridiske underpunkter:

{children_text}

Instruktioner:
- Skriv 1-2 linjer introduktion
- Derefter bullet points for hvert underpunkt
- Hver bullet max 20-25 ord (ikke karakterer)
- Bevar juridisk præcision
- Gør bullets komplette (ingen afbrudte sætninger)
- ALDRIG brug ellipser (...) eller afbryd sætninger
- Format: "• nr. X — [beskrivelse]"

Eksempel output:
Skattepligt omfatter følgende personer:
• nr. 1 — personer med bopæl i Danmark
• nr. 2 — personer med ophold over 6 måneder
"""

    try:
        response = llm.completions.create(prompt=prompt)
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Warning: LLM call failed: {e}")
        return None

def strip_md(line: str) -> str:
    """Fjern kun markdown # fra almindelige linjer."""
    return RE_MD_PREFIX.sub('', line).strip()

def strip_structural_markers(line: str, line_type: str = "body") -> str:
    """Strip kun strukturmarkeringer fra overskriftslinjer, bevar brødtekst ren."""
    if line_type == "stk":
        return re.sub(r'^###\s*stk\.\s*[0-9A-Za-z]+\.\s*', '', line, flags=re.IGNORECASE).strip()
    elif line_type == "nr":
        return re.sub(r'^###\s*\d+(?:\s*[a-zA-Z]+)?[\)\.]?\s*', '', line).strip()
    elif line_type == "litra":
        return re.sub(r'^####\s*[a-z][\)\.]?\s*', '', line, flags=re.IGNORECASE).strip()
    elif line_type == "punkt":
        return re.sub(r'^#####\s*\d+[\)\.]?\s*', '', line).strip()
    else:
        return strip_md(line)

def first_sentence(text: str, max_words: int = 28, fallback_words: int = 24) -> str:
    t = text.strip()
    dot = t.find('.')
    candidate = t[:dot+1] if 0 < dot < len(t)-1 else t
    words = candidate.split()
    if len(words) > max_words:
        candidate = ' '.join(words[:fallback_words]).rstrip(',;') + ' …'
    return candidate.strip()

def summarize_bullet(text: str, max_words: int = 50) -> str:
    t = first_sentence(text, max_words=max_words, fallback_words=max_words-5)
    if len(t.split()) < 8:
        return text[:180] + (' …' if len(text) > 180 else '')
    return t or text[:150] + (' …' if len(text) > 150 else '')

def summarize_bullet_no_ellipses(text: str, max_words: int = 25) -> str:
    t = text.strip()
    dot = t.find('.')
    if 0 < dot < len(t)-1:
        first_sent = t[:dot+1].strip()
        words = first_sent.split()
        if 5 <= len(words) <= max_words:
            return first_sent
        elif len(words) > max_words:
            return ' '.join(words[:max_words])
    
    words = t.split()
    if len(words) <= max_words:
        return t
    
    trimmed = ' '.join(words[:max_words])
    if trimmed.endswith('.'): return trimmed
    last_dot = trimmed.rfind('.')
    if last_dot > len(trimmed) * 0.7:
        return trimmed[:last_dot+1]
    
    return trimmed

def intro_no_ellipses(text: str) -> str:
    if not text: return ""
    t = text.strip()
    dot = t.find('.')
    if 0 < dot < len(t)-1:
        first_sent = t[:dot+1].strip()
        if len(first_sent.split()) >= 3:
            return first_sent
    if len(t) <= 150: return t
    truncated = t[:150]
    last_space = truncated.rfind(' ')
    return truncated[:last_space] if last_space > 50 else t

def clean_ellipses_from_text(text: str) -> str:
    if not text: return text
    cleaned = text.replace(' …', '').replace('…', '').replace(' ...', '').replace('...', '')
    return cleaned.rstrip(' ,;').strip()

def extract_dom_references(text: str) -> List[Dict[str, Any]]:
    """Udtræk domreferencer fra tekst og returnér som metadata."""
    patterns = [
        r'\bSKM\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?\d+)? ?[A-ZÆØÅ]+\b',
        r'\bTfS\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?(?:[ \.]?[A-ZÆØÅ]+)?',
        r'\bU\.? ?(?:\d{4}[\.-])? ?\d+(?:[\.\/]\d+)?(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?',
        r'\bLSR\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?(?:[ \.]?[A-ZÆØÅ]+)?',
        r'\b(?:Vestre|Østre|Højesterets)\.? ?[Ll]andsrets? ?[Dd]om af \d{1,2}\.? ?[a-zæøå]+ \d{4}\b',
        r'\b(?:Højesterets|HR)\.? ?[Dd]om af \d{1,2}\.? ?[a-zæøå]+ \d{4}\b'
    ]
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    dom_refs = []
    for match in combined_pattern.finditer(text):
        raw = match.group(0)
        norm_id = re.sub(r'\s+', '', raw).replace("/", ".").replace(" ", "").lower()
        dom_refs.append({"raw": raw, "start": match.start(), "len": len(raw), "norm_id": norm_id})
    return dom_refs

def extract_jv_references(text: str) -> List[Dict[str, Any]]:
    """Udtræk JV-referencer fra tekst og returnér som metadata."""
    patterns = [
        r'\bDen juridiske vejledning \d{4}-\d+ - [A-Z]\.[A-Z](?:\.[A-Z])* [^\n\r]+',
        r'\bVejledning om Den juridiske vejledning \d{4}-\d+',
        r'\bJV\.? ?\d{4}-\d+ [A-Z]\.[A-Z](?:\.[A-Z])*(?:\s+[^\n\r,\.;]+)?',
        r'\bDen juridiske vejledning,? ?\d{4}-\d+,? ?[A-Z]\.[A-Z](?:\.[A-Z])*(?:\s+[^\n\r,\.;]+)?',
        r'\bJV\.? ?\d{4}-\d+(?:\s+[A-Z]\.[A-Z](?:\.[A-Z])*)?(?:\s+[^\n\r,\.;]+)?',
        r'\bJV\s+afsnit\s+[A-Z]\.[A-Z](?:\.\d+)*\.?'
    ]
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    jv_refs = []
    for match in combined_pattern.finditer(text):
        raw = match.group(0).strip()
        jv_id = re.sub(r'\s+', '_', raw).lower()
        jv_id = re.sub(r'[^\w\-_\.]', '', jv_id)
        section, link_url = None, ""
        year_match = re.search(r'(\d{4})-(\d+)', raw)
        section_match = re.search(r'([A-Z]\.[A-Z](?:\.[A-Z])*)', raw, re.IGNORECASE)
        if year_match and section_match:
            year, version, section = year_match.group(1), year_match.group(2), section_match.group(1).upper()
            link_url = f"https://skat.dk/display.aspx?oid={year}-{version}-{section}"
        else:
            afsnit_match = re.search(r'afsnit\s+([A-Z]\.[A-Z](?:\.\d+)*)', raw, re.IGNORECASE)
            if afsnit_match:
                section = afsnit_match.group(1).upper()
        jv_refs.append({"raw": raw, "start": match.start(), "len": len(raw), "jv_id": jv_id, "section": section, "link": link_url})
    return jv_refs

def extract_note_anchors(original_text: str, clean_text: str) -> List[Dict[str, Any]]:
    """Udtræk note-ankre fra original tekst og juster positioner til ren tekst."""
    note_anchors = []
    matches = list(RE_INLINE_NOTE.finditer(original_text))
    for match in matches:
        note_id = match.group(1)
        original_pos = match.start()
        text_before = original_text[:original_pos]
        removed_chars = len(text_before) - len(re.sub(r'\(\u200B?\d+\u200B?\)', '', text_before))
        adjusted_pos = max(0, min(original_pos - removed_chars, len(clean_text)))
        note_anchors.append({"note_id": note_id, "start": adjusted_pos, "len": 0})
    return note_anchors

def auto_derive_law_id(filename: str) -> str:
    """Auto-aflede law_id fra filnavn."""
    filename = filename.strip()
    date_nr_match = re.search(r'\((\d{4}-\d{2}-\d{2})\s+nr\.\s*(\d+)\)', filename)
    suffix = f"{date_nr_match.group(1)}_nr{date_nr_match.group(2)}" if date_nr_match else "unknown"
    filename_lower = filename.lower()
    prefixes = {"kildeskat": "KSL", "personskat": "PSL", "selskabsskat": "SSL", "moms": "ML", "afskrivning": "AL", "ligning": "LL", "skatteforvaltning": "SFL"}
    for key, prefix in prefixes.items():
        if key in filename_lower:
            return f"{prefix}_{suffix}"
    first_word = filename.split()[0] if filename.split() else "UNK"
    prefix = re.sub(r'[^A-Za-z]', '', first_word)[:3].upper() or "UNK"
    return f"{prefix}_{suffix}"

def normalize_coordinate(value: str) -> str:
    """Normalisér koordinat-værdi til atom_id format."""
    return re.sub(r'[^\w]', '', value).lower() if value else ""

def make_atom_base_id(**kwargs) -> str:
    """Generér base atom_id uden part-suffix."""
    kwargs['part_info'] = None
    return make_atom_id(**kwargs)

def make_atom_id(law_id: str, paragraf: Optional[str] = None, stk: Optional[str] = None, 
                 nr: Optional[str] = None, lit: Optional[str] = None, pkt: Optional[str] = None,
                 part_info: Optional[tuple] = None, kind: str = "rule", 
                 note_id: Optional[str] = None, section: Optional[str] = None, 
                 root_id: Optional[str] = None) -> str:
    """Generér deterministisk atom_id."""
    parts = [law_id]
    
    if section and not paragraf:
        parts.append(f"section{normalize_coordinate(section.replace(' ', '_'))}")
    elif root_id:
        parts.append(f"root{root_id}")
    elif paragraf:
        parts.append(f"par{normalize_coordinate(paragraf)}")
        eff_stk = stk
        if stk is None and (nr or lit or pkt or kind in ["rule", "definition"]):
            eff_stk = "1"
        if eff_stk:
            parts.append(f"stk{normalize_coordinate(eff_stk)}")
        if nr: parts.append(f"nr{normalize_coordinate(nr)}")
        if lit: parts.append(f"lit{normalize_coordinate(lit)}")
        if pkt: parts.append(f"pkt{normalize_coordinate(pkt)}")

    if part_info and len(part_info) == 2:
        parts.append(f"part{part_info[0]}-of-{part_info[1]}")
    
    parts.append(f"kind{kind}")
    
    if note_id:
        parts.append(f"note{note_id}")
    
    return "--".join(filter(None, parts))

def make_chunk_id(paragraf: Optional[str], stk: Optional[str] = None, nr: Optional[str] = None,
                  lit: Optional[str] = None, pkt: Optional[str] = None,
                  *, section: Optional[str] = None, parent_label: Optional[str] = None, 
                  root_id: Optional[str] = None) -> str:
    if section:
        return f"{section}{' (parent)' if parent_label else ''}"
    
    parts = []
    if paragraf: parts.append(f"§ {paragraf}")
    if parent_label == 'paragraf_parent': return f"{', '.join(parts)} (parent)" if parts else 'root'
    
    if stk: parts.append(f"stk. {stk}")
    if parent_label == 'stk_parent': return f"{', '.join(parts)} (parent)" if parts else 'root'
    
    if nr: parts.append(f"nr. {nr}")
    if lit: parts.append(f"lit. {lit}")
    if pkt: parts.append(f"pkt. {pkt}")
    
    if not parts:
        return f"root-{root_id}" if root_id else 'root'
    
    return ', '.join(parts)

def parse_document(lines: List[str], law_id: str) -> Tuple[List[Dict[str, Any]], Dict[str, str], Dict[str, str]]:
    """Returnér (chunks, note_bodies, note_uuid_map)."""
    note_bodies, cur_id, cur_buf = {}, None, []
    root_counter = 1

    def flush_note() -> None:
        nonlocal cur_id, cur_buf
        if cur_id: note_bodies[cur_id] = ' '.join(cur_buf).strip()
        cur_id, cur_buf = None, []

    for ln in lines:
        m = RE_NOTE_BODY.match(ln)
        if m:
            flush_note()
            cur_id = m.group(1)
            cur_buf = [ln[m.end():]]
        elif cur_id:
            cur_buf.append(ln)
    flush_note()

    chunks: List[Dict[str, Any]] = []
    ctx = {"section": None, "paragraf": None, "stk": None, "nr": None, "lit": None, "pkt": None, "buf": []}

    def flush_chunk() -> None:
        nonlocal root_counter
        body = ' '.join(ctx["buf"]).strip()
        if not body:
            ctx["buf"] = []; return
        
        is_root = not ctx["paragraf"] and not ctx["section"]
        if is_root and (body.startswith('---') or 'source_file:' in body):
            print(f"🧹 Skipping YAML/metadata chunk: {body[:50]}...")
            sys.stdout.flush()
            ctx["buf"] = []; return
        
        level, kind, current_root_id = "rule", "rule", None
        if is_root:
            level, kind, current_root_id = "root", "root", str(root_counter)
            print(f"📝 FLUSH: Processing root chunk #{current_root_id}: {body[:50]}...")
            sys.stdout.flush()
            root_counter += 1
        else:
            levels = [("pkt", ctx["pkt"]), ("lit", ctx["lit"]), ("nr", ctx["nr"]), ("stk", ctx["stk"]), ("paragraf", ctx["paragraf"])]
            level = next((lvl for lvl, val in levels if val is not None), "paragraf")

        text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', body).strip()
        
        common_metadata = {
            "note_anchors": extract_note_anchors(body, text_plain),
            "dom_refs": extract_dom_references(text_plain),
            "jv_refs": extract_jv_references(text_plain),
        }
        common_metadata["note_refs"] = [a["note_id"] for a in common_metadata["note_anchors"]]

        eff_stk = ctx["stk"]
        if ctx["paragraf"] and not eff_stk and (ctx["nr"] or ctx["lit"] or ctx["pkt"]):
            eff_stk = "1"
        
        atom_kwargs = {
            "law_id": law_id, "paragraf": ctx["paragraf"], "stk": eff_stk,
            "nr": ctx["nr"], "lit": ctx["lit"], "pkt": ctx["pkt"],
            "kind": kind, "root_id": current_root_id
        }
        
        chunk_data = {
            "uuid": generate_chunk_uuid(),
            "chunk_id": make_chunk_id(**ctx, stk=eff_stk, root_id=current_root_id),
            "atom_id": make_atom_id(**atom_kwargs),
            "atom_base_id": make_atom_base_id(**atom_kwargs),
            "kind": kind, "level": level, "text_plain": text_plain,
            "note_uuids": [], "part_index": 1, "part_total": 1, "split_reason": None,
            **ctx, "stk": eff_stk, **common_metadata
        }
        
        print(f"-> CREATING: chunk_id='{chunk_data['chunk_id']}', atom_id='{chunk_data['atom_id']}'")
        sys.stdout.flush()
        chunks.append(chunk_data)
        ctx["buf"] = []

    for raw in lines:
        if raw.strip().startswith("---") and not ctx["paragraf"]: continue
        
        match_found = False
        patterns = [
            (RE_SECTION, lambda m: ctx.update({"section": f"{m.group(1)} {m.group(2)}", "paragraf": None, "stk": None, "nr": None, "lit": None, "pkt": None})),
            (RE_PARAGRAF, lambda m: (ctx.update({"paragraf": m.group(1), "stk": "1" if m.group(2).strip() else None, "nr": None, "lit": None, "pkt": None, "buf": [strip_structural_markers(m.group(2))] if m.group(2).strip() else []}))),
            (RE_STK, lambda m: ctx.update({"stk": m.group(1), "nr": None, "lit": None, "pkt": None, "buf": [strip_structural_markers(raw, "stk")]})),
            (RE_NR, lambda m: ctx.update({"nr": m.group(1), "lit": None, "pkt": None, "buf": [strip_structural_markers(raw, "nr")]}) or (not ctx["stk"] and ctx.update({"stk": "1"}))),
            (RE_LITRA, lambda m: ctx.update({"lit": m.group(1).lower(), "pkt": None, "buf": [strip_structural_markers(raw, "litra")]}) or (not ctx["nr"] and ctx.update({"nr": "1"})) or (not ctx["stk"] and ctx.update({"stk": "1"}))),
            (RE_PUNKT, lambda m: ctx.update({"pkt": m.group(1), "buf": [strip_structural_markers(raw, "punkt")]}) or (not ctx["lit"] and ctx.update({"lit": "a"})) or (not ctx["nr"] and ctx.update({"nr": "1"})) or (not ctx["stk"] and ctx.update({"stk": "1"})))
        ]
        
        for pattern, action in patterns:
            m = pattern.match(raw)
            if m and (pattern != RE_SECTION or not raw.startswith('##')):
                flush_chunk()
                action(m)
                if pattern == RE_SECTION:
                    flush_chunk()
                match_found = True
                break
        
        if not match_found:
            ctx["buf"].append(strip_structural_markers(raw))

    flush_chunk()

    note_uuid_map = {}
    for nid, ntext in note_bodies.items():
        note_uuid = generate_chunk_uuid()
        note_uuid_map[nid] = note_uuid
        text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', ntext).strip()
        atom_kwargs = {"law_id": law_id, "kind": "note", "note_id": nid}
        chunks.append({
            "uuid": note_uuid, "chunk_id": f"note({nid})",
            "atom_id": make_atom_id(**atom_kwargs), "atom_base_id": make_atom_base_id(**atom_kwargs),
            "kind": "note", "level": "note", "text_plain": text_plain,
            "section_label": None, "paragraf": None, "stk": None, "nr": None, "lit": None, "pkt": None,
            "note_anchors": extract_note_anchors(ntext, text_plain),
            "dom_refs": extract_dom_references(text_plain), "jv_refs": extract_jv_references(text_plain),
            "referenced_by": [], "note_id": int(nid),
            "part_index": 1, "part_total": 1, "split_reason": None,
        })
    return chunks, note_bodies, note_uuid_map


def build_cross_references(chunks: List[Dict[str, Any]], note_uuid_map: Dict[str, str]) -> None:
    """Build cross-references between chunks and notes."""
    note_references = {}
    for chunk in chunks:
        chunk_uuid = chunk.get("uuid")
        note_refs = [a["note_id"] for a in chunk.get("note_anchors", [])]
        chunk["note_refs"] = note_refs
        chunk["note_uuids"] = [note_uuid_map[nid] for nid in note_refs if nid in note_uuid_map]
        for note_id in note_refs:
            if note_id in note_uuid_map:
                if note_id not in note_references: note_references[note_id] = []
                note_references[note_id].append(chunk_uuid)
    
    for chunk in chunks:
        if chunk.get("level") == "note":
            note_id = str(chunk.get("note_id"))
            chunk["referenced_by"] = note_references.get(note_id, [])

def apply_note_context(chunks: List[Dict[str, Any]]) -> None:
    """Apply context from first referencing chunk to note chunks."""
    chunk_by_uuid = {chunk["uuid"]: chunk for chunk in chunks}
    for chunk in chunks:
        if chunk.get("level") == "note" and chunk.get("referenced_by"):
            parent_chunk = chunk_by_uuid.get(chunk["referenced_by"][0])
            if parent_chunk:
                for key in ["section_label", "paragraf", "stk", "nr", "lit", "pkt"]:
                    chunk[key] = parent_chunk.get(key)

def update_note_atom_ids_with_context(chunks: List[Dict[str, Any]], law_id: str) -> None:
    """Opdater note atom_ids til at inkludere koordinat-kontekst."""
    for chunk in chunks:
        if chunk.get("level") == "note" and chunk.get("paragraf"):
            atom_kwargs = {
                "law_id": law_id, "kind": "note", "note_id": str(chunk.get("note_id")),
                "paragraf": chunk.get("paragraf"), "stk": chunk.get("stk"), "nr": chunk.get("nr"),
                "lit": chunk.get("lit"), "pkt": chunk.get("pkt")
            }
            new_atom_id = make_atom_id(**atom_kwargs)
            new_atom_base_id = make_atom_base_id(**atom_kwargs)
            if chunk["atom_id"] != new_atom_id:
                print(f"🔄 Note atom_id opdateret: {chunk['atom_id']} → {new_atom_id}")
                sys.stdout.flush()
                chunk["atom_id"], chunk["atom_base_id"] = new_atom_id, new_atom_base_id

def validate_atom_id_uniqueness(chunks: List[Dict[str, Any]]) -> None:
    """Validér at alle atom_id'er er unikke."""
    from collections import Counter
    atom_ids = [c["atom_id"] for c in chunks if "atom_id" in c]
    counts = Counter(atom_ids)
    duplicates = {id: count for id, count in counts.items() if count > 1}
    
    if duplicates:
        print("❌ KRITISK FEJL: Dublerede atom_id'er fundet!")
        sys.stdout.flush()
        for dup_id, count in duplicates.items():
            print(f"   🔴 Duplikat: {dup_id} (forekommer {count} gange)")
            matching_chunks = [c for c in chunks if c.get("atom_id") == dup_id]
            for i, chunk in enumerate(matching_chunks):
                print(f"      [{i+1}] chunk_id: '{chunk.get('chunk_id')}', level: {chunk.get('level')}, kind: {chunk.get('kind')}")
        raise ValueError(f"Atom_ID unikhedskrav overtrådt! {len(duplicates)} dublerede ID'er fundet.")

def index_by_hierarchy(chunks: List[Dict[str, Any]]):
    by_p, by_ps, by_psn, paragraf_intro, section_to_paragrafs = {}, {}, {}, {}, {}
    for c in chunks:
        p, s, n, sec, level = c.get("paragraf"), c.get("stk"), c.get("nr"), c.get("section_label"), c.get("level")
        if level == 'paragraf_intro' and p:
            paragraf_intro[p] = c
            if sec:
                if sec not in section_to_paragrafs: section_to_paragrafs[sec] = []
                section_to_paragrafs[sec].append(p)
        if p:
            if p not in by_p: by_p[p] = []
            by_p[p].append(c)
        if p and s:
            if (p, s) not in by_ps: by_ps[(p, s)] = []
            by_ps[(p, s)].append(c)
        if p and s and n:
            by_psn[(p, s, n)] = c
    return by_p, by_ps, by_psn, paragraf_intro, section_to_paragrafs

def collect_children(chunks: List[Dict[str, Any]], p: str, s: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        if c.get("paragraf") != p: continue
        if s is None:
            if c.get("level") in ("stk", "stk_parent") and c.get("nr") is None: out.append(c)
        else:
            if c.get("stk") == s and c.get("level") == 'nr': out.append(c)
    
    def natural_sort_key(chunk):
        nr, stk = chunk.get("nr"), chunk.get("stk")
        nr_num, nr_letter = (9999, "")
        if nr:
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', str(nr).strip())
            if match: nr_num, nr_letter = int(match.group(1)), match.group(2) or ""
        stk_num, stk_letter = (9999, "")
        if stk:
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', str(stk).strip())
            if match: stk_num, stk_letter = int(match.group(1)), match.group(2) or ""
        return (nr_num, nr_letter, stk_num, stk_letter)
    
    out.sort(key=natural_sort_key)
    return out

def build_anchor_map_from_children(children: List[Dict[str, Any]]) -> Dict[str, str]:
    m = {}
    for ch in children:
        for nid in ch.get("note_refs", []):
            m.setdefault(nid, ch.get("chunk_id"))
    return m

def make_stk_parent(chunks: List[Dict[str, Any]], p: str, s: str,
                    paragraf_intro_c: Optional[Dict[str, Any]], law_id: str,
                    use_llm: bool = False, llm=None) -> Optional[Dict[str, Any]]:
    nr_children = collect_children(chunks, p, s)
    if not nr_children: return None
    
    if use_llm and llm:
        text_plain = generate_llm_bullets(nr_children, "stk_parent", llm)
    if not use_llm or not llm or not text_plain:
        intro = intro_no_ellipses((paragraf_intro_c or {}).get("text_plain", "").strip()) or f"Stk. {s} indeholder underpunkter (nr.)."
        bullets = [f"nr. {ch.get('nr')} — {summarize_bullet_no_ellipses(ch.get('text_plain', ''))}" for ch in nr_children]
        text_plain = intro + "\n" + "\n".join(f"• {b}" for b in bullets)

    text_plain = clean_ellipses_from_text(text_plain)
    
    atom_kwargs = {"law_id": law_id, "paragraf": p, "stk": s, "kind": "parent"}
    return {
        "uuid": generate_chunk_uuid(),
        "chunk_id": make_chunk_id(p, s, parent_label='stk_parent'),
        "atom_id": make_atom_id(**atom_kwargs),
        "atom_base_id": make_atom_base_id(**atom_kwargs),
        "kind": "parent", "level": "stk_parent",
        "section_label": nr_children[0].get("section_label"),
        "paragraf": p, "stk": s, "nr": None, "text_plain": text_plain,
        "note_anchors": [], "dom_refs": extract_dom_references(text_plain), "jv_refs": extract_jv_references(text_plain),
        "note_refs": [], "note_uuids": [],
        "children": [c.get("chunk_id") for c in nr_children],
        "nr_index": [{"nr": c.get("nr"), "child_id": c.get("chunk_id"), "has_notes": bool(c.get("note_refs"))} for c in nr_children],
        "anchor_map": build_anchor_map_from_children(nr_children),
        "created_from": {"paragraf_intro_id": (paragraf_intro_c or {}).get("chunk_id"), "child_ids": [c.get("chunk_id") for c in nr_children]},
        "part_index": 1, "part_total": 1, "split_reason": None,
    }

def make_paragraf_parent(chunks: List[Dict[str, Any]], p: str,
                         paragraf_intro_c: Optional[Dict[str, Any]], law_id: str) -> Optional[Dict[str, Any]]:
    all_stk_candidates = [c for c in chunks if c.get("paragraf") == p and c.get("level") in ("stk_parent", "stk") and c.get("nr") is None]
    if not all_stk_candidates: return None
    
    stk_children, stk_seen = [], set()
    for c in sorted(all_stk_candidates, key=lambda x: 1 if x.get("level") == "stk" else 0):
        stk_num = c.get("stk")
        if stk_num not in stk_seen:
            stk_children.append(c)
            stk_seen.add(stk_num)

    def natural_sort_key_for_stk(chunk):
        stk = chunk.get("stk")
        stk_num, stk_letter = (9999, "")
        if stk:
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', str(stk).strip())
            if match: stk_num, stk_letter = int(match.group(1)), match.group(2) or ""
        return (stk_num, stk_letter)
    stk_children.sort(key=natural_sort_key_for_stk)

    intro = intro_no_ellipses((paragraf_intro_c or {}).get("text_plain", "").strip()) or f"§ {p} indeholder følgende stykker:"
    bullets = [f"stk. {ch.get('stk')} — {summarize_bullet_no_ellipses(ch.get('text_plain', ''))}" for ch in stk_children]
    text_plain = clean_ellipses_from_text(intro + "\n" + "\n".join(f"• {b}" for b in bullets))
    
    anchor_map = {}
    for ch in stk_children:
        nr_children = collect_children(chunks, p, ch.get("stk"))
        source_children = nr_children if nr_children else [ch]
        for nid, cid in build_anchor_map_from_children(source_children).items():
            anchor_map.setdefault(nid, cid)
    
    atom_kwargs = {"law_id": law_id, "paragraf": p, "kind": "parent"}
    return {
        "uuid": generate_chunk_uuid(),
        "chunk_id": make_chunk_id(p, parent_label='paragraf_parent'),
        "atom_id": make_atom_id(**atom_kwargs),
        "atom_base_id": make_atom_base_id(**atom_kwargs),
        "kind": "parent", "level": "paragraf_parent",
        "section_label": stk_children[0].get("section_label"),
        "paragraf": p, "stk": None, "nr": None, "text_plain": text_plain,
        "note_anchors": [], "dom_refs": extract_dom_references(text_plain), "jv_refs": extract_jv_references(text_plain),
        "note_refs": [], "note_uuids": [],
        "children": [c.get("chunk_id") for c in stk_children],
        "anchor_map": anchor_map,
        "created_from": {"paragraf_intro_id": (paragraf_intro_c or {}).get("chunk_id"), "child_ids": [c.get("chunk_id") for c in stk_children]},
        "part_index": 1, "part_total": 1, "split_reason": None,
    }

def make_section_parent(section_label: str, chunks: List[Dict[str, Any]],
                        paragrafs_in_section: List[str],
                        paragraf_intro: Dict[str, Dict[str, Any]], law_id: str) -> Optional[Dict[str, Any]]:
    if not paragrafs_in_section: return None
    
    bullets, children_ids = [], []
    for p in paragrafs_in_section:
        pi = paragraf_intro.get(p)
        if not pi: continue
        bullets.append(f"§ {p} — {summarize_bullet_no_ellipses(pi.get('text_plain', ''))}")
        pp = next((c for c in chunks if c.get("paragraf") == p and c.get("level") == 'paragraf_parent'), None)
        children_ids.append(pp.get("chunk_id") if pp else pi.get("chunk_id"))
    
    if not bullets: return None
    
    intro = intro_no_ellipses(section_label)
    text_plain = clean_ellipses_from_text(intro + "\n" + "\n".join(f"• {b}" for b in bullets))
    
    anchor_map = {}
    for p in paragrafs_in_section:
        pp = next((c for c in chunks if c.get("paragraf") == p and c.get("level") == 'paragraf_parent'), None)
        if pp:
            for nid, cid in (pp.get("anchor_map") or {}).items(): anchor_map.setdefault(nid, cid)
        else:
            for ch in chunks:
                if ch.get("paragraf") == p and ch.get("level") in ("nr", "stk"):
                    for nid in ch.get("note_refs", []): anchor_map.setdefault(nid, ch.get("chunk_id"))
    
    atom_kwargs = {"law_id": law_id, "section": section_label, "kind": "parent"}
    return {
        "uuid": generate_chunk_uuid(),
        "chunk_id": make_chunk_id(None, section=section_label, parent_label='section_parent'),
        "atom_id": make_atom_id(**atom_kwargs),
        "atom_base_id": make_atom_base_id(**atom_kwargs),
        "kind": "parent", "level": "section_parent",
        "section_label": section_label,
        "paragraf": None, "stk": None, "nr": None, "text_plain": text_plain,
        "note_anchors": [], "dom_refs": extract_dom_references(text_plain), "jv_refs": extract_jv_references(text_plain),
        "note_refs": [], "note_uuids": [],
        "children": children_ids, "anchor_map": anchor_map,
        "created_from": {"child_ids": children_ids},
        "part_index": 1, "part_total": 1, "split_reason": None,
    }

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

def main():
    print(f"--- SCRIPT START: {os.path.abspath(__file__)} ---")
    sys.stdout.flush()
    
    ap = argparse.ArgumentParser(description='Markdown → juridiske chunks')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-prefix', default=None)
    ap.add_argument('--max-tokens', type=int, default=275)
    ap.add_argument('--law-id', default=None)
    ap.add_argument('--use-llm-parents', action='store_true')
    args = ap.parse_args()

    in_path = args.input
    out_prefix = args.out_prefix or os.path.splitext(os.path.basename(in_path))[0].replace(' ', '_')
    
    law_id = args.law_id or auto_derive_law_id(os.path.basename(in_path))
    print(f"📋 Law ID: {law_id}")
    sys.stdout.flush()

    with open(in_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    print("🚀 STARTING: parse_document...")
    sys.stdout.flush()
    chunks, _, note_uuid_map = parse_document(lines, law_id)
    print(f"✅ COMPLETED: parse_document - Generated {len(chunks)} initial chunks")
    sys.stdout.flush()
    
    build_cross_references(chunks, note_uuid_map)
    apply_note_context(chunks)
    update_note_atom_ids_with_context(chunks, law_id)

    print("🏗️  STARTING: Parent generation...")
    sys.stdout.flush()
    by_p, by_ps, _, p_intro, sec_to_p = index_by_hierarchy(chunks)
    
    llm = get_fireworks_llm() if args.use_llm_parents else None
    
    stk_parents = [p for p_s in by_ps for p in [make_stk_parent(chunks, p_s[0], p_s[1], p_intro.get(p_s[0]), law_id, args.use_llm_parents, llm)] if p]
    chunks.extend(stk_parents)
    
    paragraf_parents = [p for p_key in by_p for p in [make_paragraf_parent(chunks, p_key, p_intro.get(p_key), law_id)] if p]
    chunks.extend(paragraf_parents)

    section_parents = [p for sec, pars in sec_to_p.items() for p in [make_section_parent(sec, chunks, pars, p_intro, law_id)] if p]
    chunks.extend(section_parents)
    print(f"✅ COMPLETED: Parent generation. Total chunks: {len(chunks)}")
    sys.stdout.flush()


    print(f"📊 Validating {len(chunks)} chunks before splitting...")
    sys.stdout.flush()
    validate_atom_id_uniqueness(chunks)
    
    if args.max_tokens and args.max_tokens > 0:
        print(f"🔧 Splitting chunks with max_tokens={args.max_tokens}...")
        sys.stdout.flush()
        split_chunks = [part for chunk in chunks for part in split_chunk_by_tokens(chunk, args.max_tokens)]
        print(f"📊 Validating {len(split_chunks)} chunks after splitting...")
        sys.stdout.flush()
        validate_atom_id_uniqueness(split_chunks)
        chunks = split_chunks

    output_dir = os.path.dirname(out_prefix)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
        
    jsonl_path, json_path = f"{out_prefix}_chunks.jsonl", f"{out_prefix}_chunks.json"
    write_jsonl(jsonl_path, chunks)
    write_json(json_path, chunks)
    print(f"📁 Wrote {len(chunks)} chunks → {jsonl_path} + {json_path}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
