
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Legal markdown chunker med parent‑noder (stk_parent, paragraf_parent, section_parent).

Kørsel:
  python chunker.py --input "Kildeskatteloven (2025-04-11 nr. 460)_with_markdown.md" \
                    --out-prefix ksl

Output:
  ksl_chunks.jsonl
  ksl_chunks.csv

Afhænger kun af standardbiblioteket. Hvis pandas er installeret, bruges det til CSV; ellers falder scriptet tilbage til csv‑modulet.
"""

from __future__ import annotations
import re, os, json, argparse, csv, uuid
from typing import List, Dict, Any, Optional, Tuple

def safe_print(text: str):
    try:
        print(text)
    except Exception:
        try:
            import sys
            enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
            sys.stdout.write((str(text) + "\n").encode(enc, errors='replace').decode(enc))
        except Exception:
            pass

# LLM imports (optional)
try:
    from fireworks import LLM
    FIREWORKS_AVAILABLE = True
except ImportError:
    FIREWORKS_AVAILABLE = False

# ------------------------------
# Regex‑konfiguration (kan tilpasses)
# ------------------------------
# FIX: Specifik til juridiske dokumenter - undgå falske "# Indledning" osv.
RE_SECTION      = re.compile(r'^#\s*(AFSNIT|KAPITEL|DEL|BILAG)\s+(.+)$', re.IGNORECASE)
# FIX: Understøt § med bogstav og mellemrum (§ 33 A)
RE_PARAGRAF     = re.compile(r'^##\s*§\s*([0-9]+(?:\s*[A-Za-z]+)?)\.\s*(.*)$', re.IGNORECASE)
# FIX: Case-insensitive Stk. matching
RE_STK          = re.compile(r'^###\s*stk\.\s*([0-9A-Za-z]+)\.', re.IGNORECASE)
# FIX: Understøt nr. med bogstav-suffiks (nr. 10 a)
RE_NR           = re.compile(r'^###\s*(\d+(?:\s*[a-zA-Z]+)?)[\)\.]', re.IGNORECASE)
RE_NOTE_BODY    = re.compile(r'^\(\u200B?(\d+)\u200B?\)\s')   # note‑krop (med zero-width spaces)
RE_MD_PREFIX    = re.compile(r'^(#+)\s*')

RE_INLINE_NOTE  = re.compile(r'\(\u200B?(\d+)\u200B?\)')  # inline notehenvisninger (med zero-width spaces)

ANCHOR_FMT = "⟦{id}⟧"

def generate_chunk_uuid() -> str:
    """Generate full random UUID for chunks to avoid collisions."""
    return str(uuid.uuid4())  # Fuld UUID: "123e4567-e89b-12d3-a456-426614174000"


def estimate_tokens(text: str) -> int:
    """Simpel token-estimering (ca. 4 karakterer per token for dansk tekst)."""
    if not text:
        return 0
    # Simpel heuristik: 4 karakterer ≈ 1 token for dansk tekst
    # Dette kan senere erstattes med en rigtig tokenizer
    return len(text) // 4


def should_split_chunk(text: str, max_tokens: Optional[int]) -> bool:
    """Afgør om en chunk skal splittes baseret på token-begrænsning."""
    if max_tokens is None:
        return False
    return estimate_tokens(text) > max_tokens


def find_split_points(text: str, max_tokens: int) -> List[int]:
    """Find gode split-punkter i tekst baseret på sætninger og afsnit."""
    if estimate_tokens(text) <= max_tokens:
        return []
    
    # Split-strategi: Prioriter naturlige pauser
    split_points = []
    
    # 1. Prøv at splitte på afsnit (dobbelt newline)
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
        
        current_pos += len(para) + (2 if i < len(paragraphs) - 1 else 0)  # +2 for \n\n
    
    # 2. Hvis afsnit er for store, split på sætninger
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
            
            current_pos += len(sent) + (2 if i < len(sentences) - 1 else 0)  # +2 for '. '
    
    return split_points


def redistribute_metadata(original_chunk: Dict[str, Any], text_parts: List[str]) -> List[Dict[str, Any]]:
    """Fordel metadata (note_anchors, dom_refs, jv_refs) mellem split-dele."""
    parts = []
    current_pos = 0
    
    for i, text_part in enumerate(text_parts):
        part_start = current_pos
        part_end = current_pos + len(text_part)
        
        # Kopier basis chunk-struktur
        part_chunk = original_chunk.copy()
        part_chunk["uuid"] = generate_chunk_uuid()
        part_chunk["text_plain"] = text_part
        part_chunk["part_index"] = i + 1
        part_chunk["part_total"] = len(text_parts)
        part_chunk["split_reason"] = "token_limit"
        
        # Opdater chunk_id med part-info
        original_id = part_chunk.get("chunk_id", "")
        part_chunk["chunk_id"] = f"{original_id} (part {i+1})"
        
        # Filtrer metadata baseret på position
        def filter_metadata_by_position(metadata_list: List[Dict], start: int, end: int) -> List[Dict]:
            filtered = []
            for item in metadata_list:
                item_start = item.get("start", 0)
                if start <= item_start < end:
                    # Juster position relativt til denne del
                    new_item = item.copy()
                    new_item["start"] = item_start - start
                    filtered.append(new_item)
            return filtered
        
        # Fordel metadata (note_anchors skal genberegnes for split-dele)
        # For nu sættes de til tom - kan implementeres senere når split aktiveres
        part_chunk["note_anchors"] = []
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
    """Split en chunk i mindre dele baseret på token-begrænsning.
    
    FRAMEWORK: Klar til aktivering når token-begrænsning sættes.
    """
    text = chunk.get("text_plain", "")
    
    # Tjek om split er nødvendigt
    if not should_split_chunk(text, max_tokens):
        return [chunk]
    
    # PLACEHOLDER: Deaktiveret indtil videre
    safe_print(f"Split-funktionalitet er forberedt men ikke aktiveret. Chunk {chunk.get('chunk_id')} har {estimate_tokens(text)} tokens.")
    return [chunk]
    
    # FRAMEWORK KLAR TIL AKTIVERING:
    # split_points = find_split_points(text, max_tokens)
    # 
    # if not split_points:
    #     return [chunk]
    # 
    # # Split teksten
    # text_parts = []
    # last_pos = 0
    # 
    # for split_pos in split_points:
    #     text_parts.append(text[last_pos:split_pos].strip())
    #     last_pos = split_pos
    # 
    # # Tilføj sidste del
    # text_parts.append(text[last_pos:].strip())
    # 
    # # Fjern tomme dele
    # text_parts = [part for part in text_parts if part.strip()]
    # 
    # # Generer split-chunks med korrekt metadata-fordeling
    # return redistribute_metadata(chunk, text_parts)

# ------------------------------
# LLM Integration
# ------------------------------

def get_fireworks_llm(api_key=None):
    """Initialize Fireworks LLM if available."""
    if not FIREWORKS_AVAILABLE:
        return None
    try:
        # Set API key if provided
        if api_key:
            # Debug: Setting API key (removed for production)
            import fireworks
            fireworks.client.api_key = api_key
            # Also try setting as environment variable
            import os
            os.environ['FIREWORKS_API_KEY'] = api_key
        
        return LLM(
            model="qwen3-235b-a22b",
            deployment_type="serverless"
        )
    except Exception as e:
        safe_print(f"Warning: Could not initialize Fireworks LLM: {e}")
        return None

def generate_llm_bullets(children_chunks: List[Dict[str, Any]], 
                        parent_type: str = "stk_parent",
                        llm=None) -> str:
    """Generate parent chunk bullets using LLM."""
    if llm is None:
        return None
    
    # Format children for LLM prompt
    child_texts = []
    for i, child in enumerate(children_chunks, 1):
        text = child.get('text_plain', '')[:200]  # Limit text length
        child_texts.append(f"{i}. {text}")
    
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
        result = response.choices[0].text.strip()
        # DEBUG: print(f"🤖 LLM Response ({len(result)} chars): {result[:100]}...")
        return result
    except Exception as e:
        safe_print(f"Warning: LLM call failed: {e}")
        return None

# ------------------------------
# Hjælpere
# ------------------------------

def strip_md(line: str) -> str:
    """Fjern kun markdown # fra almindelige linjer."""
    return RE_MD_PREFIX.sub('', line).strip()


def strip_structural_markers(line: str, line_type: str = "body") -> str:
    """Strip kun strukturmarkeringer fra overskriftslinjer, bevar brødtekst ren."""
    if line_type == "stk":
        # Fjern: ### Stk. 2. → bevar resten
        return re.sub(r'^###\s*stk\.\s*[0-9A-Za-z]+\.\s*', '', line, flags=re.IGNORECASE).strip()
    elif line_type == "nr":
        # Fjern: ### 10 a) → bevar resten
        return re.sub(r'^###\s*\d+(?:\s*[a-zA-Z]+)?[\)\.]?\s*', '', line).strip()
    else:
        # Brødtekst: fjern kun markdown #
        return strip_md(line)





def first_sentence(text: str, max_words: int = 28, fallback_words: int = 24) -> str:
    """Enkel heuristik til kort essens: tag første sætning eller truncér på ordgrænse."""
    t = text.strip()
    # find første punktum
    dot = t.find('.')
    if 0 < dot < len(t)-1:
        candidate = t[:dot+1]
    else:
        candidate = t
    # ord‑begrænsning
    words = candidate.split()
    if len(words) > max_words:
        candidate = ' '.join(words[:fallback_words]).rstrip(',;') + ' …'
    return candidate.strip()


def summarize_bullet(text: str, max_words: int = 50) -> str:
    t = first_sentence(text, max_words=max_words, fallback_words=max_words-5)
    # Hvis første sætning er for kort, tag mere kontekst
    if len(t.split()) < 8:  # Hvis under 8 ord, tag mere
        return text[:180] + (' …' if len(text) > 180 else '')
    # fald tilbage hvis tom
    return t or text[:150] + (' …' if len(text) > 150 else '')


def summarize_bullet_no_ellipses(text: str, max_words: int = 25) -> str:
    """Summarize bullet for parent chunks WITHOUT ellipses - keep complete sentences.
    Enforces word limit for consistency with LLM output (20-25 words)."""
    t = text.strip()
    
    # Find første komplet sætning
    dot = t.find('.')
    if 0 < dot < len(t)-1:
        first_sent = t[:dot+1].strip()
        words = first_sent.split()
        
        # Tjek ordgrænse på første sætning
        if len(words) >= 5 and len(words) <= max_words:
            return first_sent
        elif len(words) > max_words:
            # Trim til max_words på ordgrænse
            return ' '.join(words[:max_words])
    
    # Hvis ingen god første sætning, trim hele teksten til ordgrænse
    words = t.split()
    if len(words) <= max_words:
        return t
    
    # Trim til max_words og prøv at ende på punktum hvis muligt
    trimmed = ' '.join(words[:max_words])
    if trimmed.endswith('.'):
        return trimmed
    
    # Find sidste punktum inden for trimmed tekst
    last_dot = trimmed.rfind('.')
    if last_dot > len(trimmed) * 0.7:  # Hvis punktum er mindst 70% inde
        return trimmed[:last_dot+1]
    
    # Fallback: brug trimmed tekst uden punktum
    return trimmed


def intro_no_ellipses(text: str) -> str:
    """Generate intro for parent chunks WITHOUT ellipses - keep complete sentences."""
    if not text:
        return ""
    
    t = text.strip()
    # Find første komplet sætning
    dot = t.find('.')
    if 0 < dot < len(t)-1:
        # Tag første sætning hvis den er rimelig lang
        first_sent = t[:dot+1].strip()
        if len(first_sent.split()) >= 3:  # Mindst 3 ord
            return first_sent
    
    # Hvis ingen god første sætning, tag op til 150 karakterer uden at afbryde ord
    if len(t) <= 150:
        return t
    
    # Find sidste mellemrum inden for 150 karakterer
    truncated = t[:150]
    last_space = truncated.rfind(' ')
    if last_space > 50:  # Hvis der er et mellemrum efter 50 karakterer
        return truncated[:last_space]
    
    # Fallback: tag hele teksten
    return t


def clean_ellipses_from_text(text: str) -> str:
    """Remove all ellipses and truncation artifacts from parent text."""
    if not text:
        return text
    
    # Fjern forskellige ellipse-varianter
    cleaned = text
    cleaned = cleaned.replace(' …', '')
    cleaned = cleaned.replace('…', '')
    cleaned = cleaned.replace(' ...', '')
    cleaned = cleaned.replace('...', '')
    
    # Fjern trailing kommaer og semikolon der kan være tilbage efter ellipse-fjernelse
    cleaned = cleaned.rstrip(' ,;')
    
    return cleaned.strip()


def extract_dom_references(text: str) -> List[Dict[str, Any]]:
    """Udtræk domreferencer fra tekst og returnér som metadata."""
    # Udvidede mønstre til at fange flere formater
    patterns = [
        r'\bSKM\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?\d+)? ?[A-ZÆØÅ]+\b',  # SKM 2023 7 HR, SKM.2023.7.HR, SKM 2003 405 HR
        r'\bTfS\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?(?:[ \.]?[A-ZÆØÅ]+)?',  # TfS 1998 354 H, TfS.1998.354.H, TfS 1995 137 LSR
        r'\bU\.? ?(?:\d{4}[\.-])? ?\d+(?:[\.\/]\d+)?(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?',  # U 2004.234, U.2004/234H
        r'\bLSR\.? ?(?:\d{4}[\.-])? ?\d+(?:[ \.]?[A-ZÆØÅ]+)?(?:[ \.]?\d+)?(?:[ \.]?[A-ZÆØÅ]+)?',  # LSR 2022 42, LSR.2022.42.SR
        r'\b(?:Vestre|Østre|Højesterets)\.? ?[Ll]andsrets? ?[Dd]om af \d{1,2}\.? ?[a-zæøå]+ \d{4}\b',  # Vestre Landsrets Dom af 12. juni 2018
        r'\b(?:Højesterets|HR)\.? ?[Dd]om af \d{1,2}\.? ?[a-zæøå]+ \d{4}\b'  # Højesterets Dom af 4. marts 2022
    ]
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    
    dom_refs = []
    for match in combined_pattern.finditer(text):
        raw = match.group(0)
        norm_id = re.sub(r'\s+', '', raw).replace("/", ".").replace(" ", "").lower()
        dom_refs.append({
            "raw": raw,
            "start": match.start(),
            "len": len(raw),
            "norm_id": norm_id
        })
    
    return dom_refs


def extract_jv_references(text: str) -> List[Dict[str, Any]]:
    """Udtræk JV-referencer fra tekst og returnér som metadata."""
    # Mønstre til at fange JV-referencer - inkluderer både årstal-versioner og afsnit-formater
    patterns = [
        # Den juridiske vejledning 2025-1 - A.A Processuelle regler for Skatteforvaltningens opgaver
        r'\bDen juridiske vejledning \d{4}-\d+ - [A-Z]\.[A-Z](?:\.[A-Z])* [^\n\r]+',
        
        # Vejledning om Den juridiske vejledning 2025-1
        r'\bVejledning om Den juridiske vejledning \d{4}-\d+',
        
        # Kortere former som "JV 2025-1 A.A"
        r'\bJV\.? ?\d{4}-\d+ [A-Z]\.[A-Z](?:\.[A-Z])*(?:\s+[^\n\r,\.;]+)?',
        
        # "Den juridiske vejledning" efterfulgt af år og sektion
        r'\bDen juridiske vejledning,? ?\d{4}-\d+,? ?[A-Z]\.[A-Z](?:\.[A-Z])*(?:\s+[^\n\r,\.;]+)?',
        
        # Generisk "JV" reference med årstal
        r'\bJV\.? ?\d{4}-\d+(?:\s+[A-Z]\.[A-Z](?:\.[A-Z])*)?(?:\s+[^\n\r,\.;]+)?',
        
        # Afsnit-format fra inputfilen: "JV afsnit C.F.1.2.3"
        r'\bJV\s+afsnit\s+[A-Z]\.[A-Z](?:\.\d+)*\.?'
    ]
    
    combined_pattern = re.compile('|'.join(patterns), re.IGNORECASE)
    
    jv_refs = []
    for match in combined_pattern.finditer(text):
        raw = match.group(0).strip()
        
        # Generer et unikt ID baseret på JV-referencen
        jv_id = re.sub(r'\s+', '_', raw).lower()
        jv_id = re.sub(r'[^\w\-_\.]', '', jv_id)
        
        # Udtræk sektion information
        section = None
        link_url = ""
        
        # Tjek for årstal-version format
        year_match = re.search(r'(\d{4})-(\d+)', raw)
        section_match = re.search(r'([A-Z]\.[A-Z](?:\.[A-Z])*)', raw, re.IGNORECASE)
        
        if year_match and section_match:
            year = year_match.group(1)
            version = year_match.group(2)
            section = section_match.group(1).upper()
            link_url = f"https://skat.dk/display.aspx?oid={year}-{version}-{section}"
        else:
            # Tjek for afsnit-format: "JV afsnit C.F.1.2.3"
            afsnit_match = re.search(r'afsnit\s+([A-Z]\.[A-Z](?:\.\d+)*)', raw, re.IGNORECASE)
            if afsnit_match:
                section = afsnit_match.group(1).upper()
        
        jv_refs.append({
            "raw": raw,
            "start": match.start(),
            "len": len(raw),
            "jv_id": jv_id,
            "section": section,
            "link": link_url
        })
    
    return jv_refs


def extract_note_anchors(original_text: str, clean_text: str) -> List[Dict[str, Any]]:
    """Udtræk note-ankre fra original tekst og juster positioner til ren tekst."""
    note_anchors = []
    
    # Find alle note-markeringer i original tekst
    matches = list(RE_INLINE_NOTE.finditer(original_text))
    
    # Beregn offset-justering for hver position
    for match in matches:
        note_id = match.group(1)
        original_pos = match.start()
        
        # Beregn hvor mange karakterer der er fjernet før denne position
        text_before = original_text[:original_pos]
        removed_chars = len(text_before) - len(re.sub(r'\(\u200B?\d+\u200B?\)', '', text_before))
        
        # Justeret position i ren tekst
        adjusted_pos = original_pos - removed_chars
        
        # Sørg for at position ikke er negativ eller over grænsen
        adjusted_pos = max(0, min(adjusted_pos, len(clean_text)))
        
        note_anchors.append({
            "note_id": note_id,
            "start": adjusted_pos,
            "len": 0  # Indsætningspunkt, ikke eksisterende tekst
        })
    
    return note_anchors








def make_chunk_id(paragraf: Optional[str], stk: Optional[str] = None, nr: Optional[str] = None,
                  *, section: Optional[str] = None, parent_label: Optional[str] = None, is_intro: bool = False) -> str:
    if section is not None:
        base = f"{section}"
        if parent_label:
            return f"{base} (parent)"
        return base
    parts: List[str] = []
    if paragraf:
        parts.append(f"§ {paragraf}")
    
    # FIX: Entydige IDs for intro og parent
    if is_intro:
        return (', '.join(parts) + " (intro)") if parts else 'root'
    if parent_label == 'paragraf_parent':
        return (', '.join(parts) + " (parent)") if parts else 'root'
    
    if stk is not None:
        parts.append(f"stk. {stk}")
    if parent_label == 'stk_parent':
        return (', '.join(parts) + " (parent)") if parts else 'root'
    if nr is not None:
        parts.append(f"nr. {nr}")
    return ', '.join(parts) if parts else 'root'


# ------------------------------
# Parsing af dokument → basis‑chunks
# ------------------------------

def parse_document(lines: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """Returnér (chunks, note_bodies)."""
    # 1) Opsamling af note‑kroppe nederst
    note_bodies: Dict[str, str] = {}
    cur_id: Optional[str] = None
    cur_buf: List[str] = []
    def flush_note() -> None:
        nonlocal cur_id, cur_buf
        if cur_id is None:
            return
        note_bodies[cur_id] = ' '.join(cur_buf).strip()
        cur_id, cur_buf = None, []

    for ln in lines:
        m = RE_NOTE_BODY.match(ln)
        if m:
            flush_note()
            cur_id = m.group(1)
            cur_buf = [ln[m.end():]]
        else:
            if cur_id is not None:
                cur_buf.append(ln)
    flush_note()

    # 2) Struktur‑parse
    chunks: List[Dict[str, Any]] = []
    ctx = {"section": None, "paragraf": None, "stk": None, "nr": None, "buf": []}

    def flush_chunk() -> None:
        body = ' '.join(ctx["buf"]).strip()
        if not body:
            ctx["buf"] = []
            return
        # FIX: Skip hvis ingen aktiv struktur (undgå "root" støj fra YAML front matter)
        if not ctx["paragraf"] and not ctx["section"]:
            # Extra check: skip if content looks like YAML front matter or metadata
            if body.startswith('---') or 'source_file:' in body or 'converted_at:' in body:
                safe_print(f"Skipping YAML/metadata chunk: {body[:50]}...")
                ctx["buf"] = []
                return
            ctx["buf"] = []
            return
        # niveau
        if ctx["paragraf"] and ctx["stk"] is None and ctx["nr"] is None:
            eff_stk = "1"
            level = "stk"
        else:
            eff_stk = ctx["stk"]
            level = "nr" if ctx["nr"] is not None else ("stk" if ctx["stk"] is not None else "paragraf")
        
        # Fjern note-markeringer fra tekst for at få ren text_plain FØRST
        text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', body).strip()
        
        # Udtræk alle metadata med korrekte positioner
        note_anchors = extract_note_anchors(body, text_plain)  # Juster positioner til ren tekst
        dom_refs = extract_dom_references(text_plain)
        jv_refs = extract_jv_references(text_plain)
        
        # Udtræk note_ids for kompatibilitet
        note_ids = [anchor["note_id"] for anchor in note_anchors]
        
        chunks.append({
            "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
            "chunk_id": make_chunk_id(ctx["paragraf"], eff_stk, ctx["nr"], section=None),
            "level": level,
            "section_label": ctx["section"],
            "paragraf": ctx["paragraf"],
            "stk": eff_stk,
            "nr": ctx["nr"],
            "text_plain": text_plain,
            "note_anchors": note_anchors,
            "dom_refs": dom_refs,
            "jv_refs": jv_refs,
            "note_refs": note_ids,  # Bevar for kompatibilitet
            "note_uuids": [],  # Vil blive udfyldt senere
            # Split-ready struktur (klar til aktivering)
            "part_index": 1,
            "part_total": 1,
            "split_reason": None,
        })
        ctx["buf"] = []

    current_section_label: Optional[str] = None

    for raw in lines:
        # Afsnit/kapitel‑linje
        m_sec = RE_SECTION.match(raw)
        if m_sec and not raw.startswith('##') and not raw.startswith('###'):
            # flush forrige chunk
            flush_chunk()
            section_type, section_title = m_sec.groups()
            current_section_label = f"{section_type} {section_title}"
            ctx.update({"section": current_section_label, "paragraf": None, "stk": None, "nr": None, "buf": []})
            # selve overskriften som chunk (level=afsnit)
            # Fjern note-markeringer fra tekst for at få ren text_plain FØRST
            text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', current_section_label).strip()
            
            # Udtræk metadata med korrekte positioner
            note_anchors = extract_note_anchors(current_section_label, text_plain)
            dom_refs = extract_dom_references(text_plain)
            jv_refs = extract_jv_references(text_plain)
            
            # Udtræk note_ids for kompatibilitet
            note_ids = [anchor["note_id"] for anchor in note_anchors]
            
            chunks.append({
                "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
                "chunk_id": make_chunk_id(None, section=current_section_label),
                "level": "afsnit",
                "section_label": current_section_label,
                "paragraf": None, "stk": None, "nr": None,
                "text_plain": text_plain,
                "note_anchors": note_anchors,
                "dom_refs": dom_refs,
                "jv_refs": jv_refs,
                "note_refs": note_ids,
                "note_uuids": [],
                # Split-ready struktur (klar til aktivering)
                "part_index": 1,
                "part_total": 1,
                "split_reason": None,
            })
            continue
        
        # Paragraf‑linje m. mulig introhale
        m_p = RE_PARAGRAF.match(raw)
        if m_p:
            flush_chunk()
            pnum, tail = m_p.groups()
            # DEBUG: print(f"🔍 Found paragraf: '{pnum}' (from line: {raw[:50]}...)")
            ctx.update({"paragraf": pnum, "stk": None, "nr": None, "buf": []})
            tail = tail.strip()
            if tail:
                # paragraf_intro som eget chunk - FIX: Ren tekst uden § label
                clean_tail = strip_structural_markers(tail, "body")  # Kun fjern markdown
                
                # Fjern note-markeringer fra tekst for at få ren text_plain FØRST
                text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', clean_tail).strip()
                
                # Udtræk metadata med korrekte positioner
                note_anchors = extract_note_anchors(clean_tail, text_plain)
                dom_refs = extract_dom_references(text_plain)
                jv_refs = extract_jv_references(text_plain)
                
                # Udtræk note_ids for kompatibilitet
                note_ids = [anchor["note_id"] for anchor in note_anchors]
                
                chunks.append({
                    "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
                    "chunk_id": make_chunk_id(pnum, is_intro=True),  # FIX: Entydigt ID
                    "level": "paragraf_intro",
                    "section_label": current_section_label,
                    "paragraf": pnum, "stk": None, "nr": None,
                    "text_plain": text_plain,
                    "note_anchors": note_anchors,
                    "dom_refs": dom_refs,
                    "jv_refs": jv_refs,
                    "note_refs": note_ids,
                    "note_uuids": [],
                    # Split-ready struktur (klar til aktivering)
                    "part_index": 1,
                    "part_total": 1,
                    "split_reason": None,
                })
            continue
        
        # Stk.
        m_s = RE_STK.match(raw)
        if m_s:
            flush_chunk()
            s = m_s.group(1)
            # DEBUG: print(f"🔍 Found stk: '{s}' (from line: {raw[:50]}...)")
            # FIX: Brug strip_structural_markers for ren tekst
            clean_text = strip_structural_markers(raw, "stk")
            ctx.update({"stk": s, "nr": None, "buf": [clean_text] if clean_text else []})
            continue
        
        # Nr.
        m_n = RE_NR.match(raw)
        if m_n:
            flush_chunk()
            if ctx["stk"] is None:
                ctx["stk"] = "1"
            n = m_n.group(1)
            # DEBUG: print(f"🔍 Found nr: '{n}' (from line: {raw[:50]}...)")
            # FIX: Brug strip_structural_markers for ren tekst
            clean_text = strip_structural_markers(raw, "nr")
            ctx.update({"nr": n, "buf": [clean_text] if clean_text else []})
            continue
        
        # Akkumuler brødtekst - FIX: Brug strip_structural_markers for ren tekst
        ctx["buf"].append(strip_structural_markers(raw, "body"))

    flush_chunk()

    # Note‑chunks som selvstændige noder
    note_uuid_map = {}  # Map note_id -> uuid
    for nid, ntext in note_bodies.items():
        note_uuid = generate_chunk_uuid()
        note_uuid_map[nid] = note_uuid
        
        # Fjern note-markeringer fra tekst for at få ren text_plain FØRST
        text_plain = re.sub(r'\(\u200B?\d+\u200B?\)', '', ntext).strip()
        
        # Udtræk metadata med korrekte positioner
        note_anchors = extract_note_anchors(ntext, text_plain)
        dom_refs = extract_dom_references(text_plain)
        jv_refs = extract_jv_references(text_plain)
        
        chunks.append({
            "uuid": note_uuid,
            "chunk_id": f"note({nid})",
            "level": "note",
            "section_label": None,
            "paragraf": None, "stk": None, "nr": None,
            "text_plain": text_plain,
            "note_anchors": note_anchors,
            "dom_refs": dom_refs,
            "jv_refs": jv_refs,
            "referenced_by": [],  # Vil blive udfyldt senere
            # Split-ready struktur (klar til aktivering)
            "note_id": int(nid),  # Stabil note-id som int
            "part_index": 1,
            "part_total": 1,
            "split_reason": None,
        })

    return chunks, note_bodies, note_uuid_map


def build_cross_references(chunks: List[Dict[str, Any]], note_uuid_map: Dict[str, str]) -> None:
    """Build cross-references between chunks and notes."""
    # Map note_id -> list of chunk UUIDs that reference it
    note_references = {}
    
    for chunk in chunks:
        chunk_uuid = chunk.get("uuid")
        note_refs = chunk.get("note_refs", [])
        
        # Build note_uuids list for this chunk
        note_uuids = []
        for note_id in note_refs:
            if note_id in note_uuid_map:
                note_uuid = note_uuid_map[note_id]
                note_uuids.append(note_uuid)
                # Track reverse reference
                if note_id not in note_references:
                    note_references[note_id] = []
                note_references[note_id].append(chunk_uuid)
        
        chunk["note_uuids"] = note_uuids
    
    # Update note chunks with referenced_by information
    for chunk in chunks:
        if chunk.get("level") == "note":
            chunk_id = chunk.get("chunk_id", "")
            # Extract note_id from "note(14)" format
            if chunk_id.startswith("note(") and chunk_id.endswith(")"):
                note_id = chunk_id[5:-1]  # Remove "note(" and ")"
                chunk["referenced_by"] = note_references.get(note_id, [])


def apply_note_context(chunks: List[Dict[str, Any]]) -> None:
    """Apply context from first referencing chunk to note chunks."""
    # Create UUID lookup for quick access
    chunk_by_uuid = {chunk["uuid"]: chunk for chunk in chunks}
    
    for chunk in chunks:
        if chunk.get("level") == "note":
            referenced_by = chunk.get("referenced_by", [])
            if referenced_by:
                # Get first referencing chunk
                first_ref_uuid = referenced_by[0]
                if first_ref_uuid in chunk_by_uuid:
                    parent_chunk = chunk_by_uuid[first_ref_uuid]
                    
                    # Copy context from parent to note
                    chunk["section_label"] = parent_chunk.get("section_label")
                    chunk["paragraf"] = parent_chunk.get("paragraf") 
                    chunk["stk"] = parent_chunk.get("stk")
                    chunk["nr"] = parent_chunk.get("nr")


def clean_orphaned_note_references(chunks: List[Dict[str, Any]], note_uuid_map: Dict[str, str]) -> None:
    """Remove note references that don't have corresponding note chunks."""
    valid_note_ids = set(note_uuid_map.keys())
    
    for chunk in chunks:
        if chunk.get("level") == "note":
            continue  # Skip note chunks themselves
            
        # Clean note_refs (for kompatibilitet)
        original_note_refs = chunk.get("note_refs", [])
        cleaned_note_refs = [nid for nid in original_note_refs if nid in valid_note_ids]
        chunk["note_refs"] = cleaned_note_refs
        
        # Clean note_uuids accordingly (for kompatibilitet)
        cleaned_note_uuids = [note_uuid_map[nid] for nid in cleaned_note_refs]
        chunk["note_uuids"] = cleaned_note_uuids
        
        # Clean note_anchors metadata - fjern ugyldige note-ankre
        original_note_anchors = chunk.get("note_anchors", [])
        cleaned_note_anchors = [
            anchor for anchor in original_note_anchors 
            if anchor.get("note_id") in valid_note_ids
        ]
        chunk["note_anchors"] = cleaned_note_anchors
        
        # Report cleaning if any orphaned refs were found
        orphaned_refs = set(original_note_refs) - set(cleaned_note_refs)
        orphaned_anchors = len(original_note_anchors) - len(cleaned_note_anchors)
        
        if orphaned_refs or orphaned_anchors > 0:
            if orphaned_refs:
                safe_print(f"Cleaned orphaned note refs {orphaned_refs} from {chunk.get('chunk_id', 'unknown')}")
            if orphaned_anchors > 0:
                safe_print(f"Cleaned {orphaned_anchors} orphaned note anchors from {chunk.get('chunk_id', 'unknown')}")


# ------------------------------
# Parent‑generation
# ------------------------------

def index_by_hierarchy(chunks: List[Dict[str, Any]]):
    by_p: Dict[str, List[Dict[str, Any]]] = {}
    by_ps: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    by_psn: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    paragraf_intro: Dict[str, Dict[str, Any]] = {}
    section_to_paragrafs: Dict[str, List[str]] = {}

    for c in chunks:
        level = c.get("level")
        p, s, n = c.get("paragraf"), c.get("stk"), c.get("nr")
        sec = c.get("section_label")
        if level == 'paragraf_intro' and p:
            paragraf_intro[p] = c
            if sec:
                section_to_paragrafs.setdefault(sec, []).append(p)
        if p:
            by_p.setdefault(p, []).append(c)
        if p and s is not None:
            by_ps.setdefault((p, s), []).append(c)
        if p and s is not None and n is not None:
            by_psn[(p, s, n)] = c
    return by_p, by_ps, by_psn, paragraf_intro, section_to_paragrafs


def collect_children(chunks: List[Dict[str, Any]], p: str, s: Optional[str] = None) -> List[Dict[str, Any]]:
    out = []
    for c in chunks:
        if c.get("paragraf") != p:
            continue
        if s is None:
            # alle stykker og nr under §
            if c.get("level") in ("stk", "stk_parent") and c.get("nr") is None:
                out.append(c)
        else:
            # s er specificeret - find nr-børn under dette stykke
            if c.get("stk") == s and c.get("level") == 'nr':
                out.append(c)
    # stabil orden: efter nr/stk med naturlig sortering af tal+bogstav
    def natural_sort_key(chunk):
        nr = chunk.get("nr")
        stk = chunk.get("stk")
        
        # Parse nr (kan være "10", "10a", "10 a", etc.)
        nr_num = 999999  # fallback for ikke-numeriske
        nr_letter = ""
        if nr is not None:
            nr_str = str(nr).strip()
            # Find tal-delen
            import re
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', nr_str)
            if match:
                nr_num = int(match.group(1))
                nr_letter = match.group(2) or ""
        
        # Parse stk (kan være "1", "2a", etc.)
        stk_num = 999999  # fallback
        stk_letter = ""
        if stk is not None:
            stk_str = str(stk).strip()
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', stk_str)
            if match:
                stk_num = int(match.group(1))
                stk_letter = match.group(2) or ""
        
        return (nr_num, nr_letter, stk_num, stk_letter)
    
    out.sort(key=natural_sort_key)
    return out


def build_anchor_map_from_children(children: List[Dict[str, Any]]) -> Dict[str, str]:
    m: Dict[str, str] = {}
    for ch in children:
        for nid in ch.get("note_refs", []) or []:
            m.setdefault(nid, ch.get("chunk_id"))
    return m


def make_stk_parent(chunks: List[Dict[str, Any]], p: str, s: str,
                    paragraf_intro_c: Optional[Dict[str, Any]], 
                    use_llm: bool = False, llm=None) -> Optional[Dict[str, Any]]:
    # find nr‑børn
    nr_children = collect_children(chunks, p, s)
    if not nr_children:
        return None
    # Generate content with LLM if enabled
    if use_llm and llm:
        llm_text = generate_llm_bullets(nr_children, "stk_parent", llm)
        if llm_text:
            text_plain = llm_text
        else:
            # Fallback to rule-based
            intro = intro_no_ellipses((paragraf_intro_c or {}).get("text_plain", "").strip()) if paragraf_intro_c else ""
            intro = intro or f"Stk. {s} indeholder underpunkter (nr.)."
            bullets = []
            for ch in nr_children:
                nr = ch.get("nr")
                ess = summarize_bullet_no_ellipses(ch.get("text_plain", ""))
                bullets.append(f"nr. {nr} — {ess}")
            text_plain = intro + "\n" + "\n".join(f"• {b}" for b in bullets)
    else:
        # Rule-based generation
        intro = intro_no_ellipses((paragraf_intro_c or {}).get("text_plain", "").strip()) if paragraf_intro_c else ""
        intro = intro or f"Stk. {s} indeholder underpunkter (nr.)."
        bullets = []
        for ch in nr_children:
            nr = ch.get("nr")
            ess = summarize_bullet_no_ellipses(ch.get("text_plain", ""))
            bullets.append(f"nr. {nr} — {ess}")
        text_plain = intro + "\n" + "\n".join(f"• {b}" for b in bullets)

    children_ids = [c.get("chunk_id") for c in nr_children]
    anchor_map = build_anchor_map_from_children(nr_children)

    # FIX: Post-tjek for ellipser
    text_plain = clean_ellipses_from_text(text_plain)
    
    # Parent chunks indeholder genereret tekst uden note-markeringer
    text_plain_clean = text_plain.strip()
    
    # Udtræk metadata fra parent tekst (ingen note-ankre i genereret tekst)
    note_anchors = []  # Parent chunks har ingen note-ankre
    dom_refs = extract_dom_references(text_plain_clean)
    jv_refs = extract_jv_references(text_plain_clean)

    return {
        "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
        "chunk_id": make_chunk_id(p, s, None, parent_label='stk_parent'),
        "level": "stk_parent",
        "section_label": nr_children[0].get("section_label"),
        "paragraf": p, "stk": s, "nr": None,
        "text_plain": text_plain_clean,
        "note_anchors": note_anchors,
        "dom_refs": dom_refs,
        "jv_refs": jv_refs,
        "note_refs": [],
        "note_uuids": [],
        "children": children_ids,
        "nr_index": [{"nr": c.get("nr"), "child_id": c.get("chunk_id"), "has_notes": bool(c.get("note_refs"))} for c in nr_children],
        "anchor_map": anchor_map,
        "created_from": {"paragraf_intro_id": (paragraf_intro_c or {}).get("chunk_id"), "child_ids": children_ids},
        # Split-ready struktur (klar til aktivering)
        "part_index": 1,
        "part_total": 1,
        "split_reason": None,
    }


def make_paragraf_parent(chunks: List[Dict[str, Any]], p: str,
                         paragraf_intro_c: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    # find stk‑børn (brug stk_parent hvis den senere tilføjes; ellers stk)
    all_stk_candidates = [c for c in chunks if c.get("paragraf") == p and c.get("level") in ("stk_parent", "stk") and c.get("nr") is None]
    if not all_stk_candidates:
        return None
    
    # FIX: Deduplikér - prioriter stk_parent over stk for samme stykke
    stk_children = []
    stk_seen = set()
    
    # Først tilføj alle stk_parent chunks
    for c in all_stk_candidates:
        if c.get("level") == "stk_parent":
            stk_num = c.get("stk")
            if stk_num not in stk_seen:
                stk_children.append(c)
                stk_seen.add(stk_num)
    
    # Derefter tilføj stk chunks kun hvis der ikke allerede er en stk_parent
    for c in all_stk_candidates:
        if c.get("level") == "stk":
            stk_num = c.get("stk")
            if stk_num not in stk_seen:
                stk_children.append(c)
                stk_seen.add(stk_num)
    
    # FIX: Naturlig sortering som i collect_children - håndterer "1", "1a", "1b", "2", "2a" korrekt
    def natural_sort_key_for_stk(chunk):
        stk = chunk.get("stk")
        stk_num = 999999  # fallback
        stk_letter = ""
        if stk is not None:
            stk_str = str(stk).strip()
            import re
            match = re.match(r'^(\d+)(?:\s*([a-zA-Z]+))?', stk_str)
            if match:
                stk_num = int(match.group(1))
                stk_letter = match.group(2) or ""
        return (stk_num, stk_letter)
    stk_children.sort(key=natural_sort_key_for_stk)
    intro = intro_no_ellipses((paragraf_intro_c or {}).get("text_plain", "").strip()) or f"§ {p} indeholder følgende stykker:"
    bullets = []
    for ch in stk_children:
        s = ch.get("stk")
        # Forsøg at destillere en kort essens
        basis = ch.get("text_plain", "")
        ess = summarize_bullet_no_ellipses(basis)
        bullets.append(f"stk. {s} — {ess}")
    text_plain = intro + "\n" + "\n".join(f"• {b}" for b in bullets)

    # anchor_map: aggregation fra alle underliggende (brug laveste niveau barnet vi har)
    anchor_map: Dict[str, str] = {}
    for ch in stk_children:
        # helst mappes til nr‑børn hvis de findes
        nr_children = collect_children(chunks, p, ch.get("stk"))
        if nr_children:
            for nid, cid in build_anchor_map_from_children(nr_children).items():
                anchor_map.setdefault(nid, cid)
        # ellers til selve stk‑barnet
        for nid in ch.get("note_refs", []) or []:
            anchor_map.setdefault(nid, ch.get("chunk_id"))

    children_ids = [c.get("chunk_id") for c in stk_children]
    section_label = stk_children[0].get("section_label")

    # FIX: Post-tjek for ellipser
    text_plain = clean_ellipses_from_text(text_plain)
    
    # Parent chunks indeholder genereret tekst uden note-markeringer
    text_plain_clean = text_plain.strip()
    
    # Udtræk metadata fra parent tekst (ingen note-ankre i genereret tekst)
    note_anchors = []  # Parent chunks har ingen note-ankre
    dom_refs = extract_dom_references(text_plain_clean)
    jv_refs = extract_jv_references(text_plain_clean)

    return {
        "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
        "chunk_id": make_chunk_id(p, parent_label='paragraf_parent'),
        "level": "paragraf_parent",
        "section_label": section_label,
        "paragraf": p, "stk": None, "nr": None,
        "text_plain": text_plain_clean,
        "note_anchors": note_anchors,
        "dom_refs": dom_refs,
        "jv_refs": jv_refs,
        "note_refs": [],
        "note_uuids": [],
        "children": children_ids,
        "anchor_map": anchor_map,
        "created_from": {"paragraf_intro_id": (paragraf_intro_c or {}).get("chunk_id"), "child_ids": children_ids},
        # Split-ready struktur (klar til aktivering)
        "part_index": 1,
        "part_total": 1,
        "split_reason": None,
    }


def make_section_parent(section_label: str, chunks: List[Dict[str, Any]],
                        paragrafs_in_section: List[str],
                        paragraf_intro: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not paragrafs_in_section:
        return None
    bullets = []
    children_ids = []
    for p in paragrafs_in_section:
        pi = paragraf_intro.get(p)
        if not pi:
            continue
        ess = summarize_bullet_no_ellipses(pi.get("text_plain", ""))
        bullets.append(f"§ {p} — {ess}")
        # Child peger helst til paragraf_parent (hvis findes) ellers paragraf_intro
        pp = next((c for c in chunks if c.get("paragraf") == p and c.get("level") == 'paragraf_parent'), None)
        children_ids.append(pp.get("chunk_id") if pp else pi.get("chunk_id"))

    if not bullets:
        return None

    intro = intro_no_ellipses(section_label)  # FIX: Konsistens - brug hele sætning/label
    text_plain = intro + "\n" + "\n".join(f"• {b}" for b in bullets)

    # anchor_map: samle laveste børn
    anchor_map: Dict[str, str] = {}
    for p in paragrafs_in_section:
        # hvis paragraf_parent findes, saml derfra; ellers fra stk/nr
        pp = next((c for c in chunks if c.get("paragraf") == p and c.get("level") == 'paragraf_parent'), None)
        if pp:
            for nid, cid in (pp.get("anchor_map") or {}).items():
                anchor_map.setdefault(nid, cid)
        else:
            # Fald tilbage: find nr/stk for §
            for ch in chunks:
                if ch.get("paragraf") == p and ch.get("level") in ("nr", "stk"):
                    for nid in ch.get("note_refs", []) or []:
                        anchor_map.setdefault(nid, ch.get("chunk_id"))

    # FIX: Post-tjek for ellipser
    text_plain = clean_ellipses_from_text(text_plain)
    
    # Parent chunks indeholder genereret tekst uden note-markeringer
    text_plain_clean = text_plain.strip()
    
    # Udtræk metadata fra parent tekst (ingen note-ankre i genereret tekst)
    note_anchors = []  # Parent chunks har ingen note-ankre
    dom_refs = extract_dom_references(text_plain_clean)
    jv_refs = extract_jv_references(text_plain_clean)

    return {
        "uuid": generate_chunk_uuid(),  # FIX: Unikt UUID
        "chunk_id": make_chunk_id(None, section=section_label, parent_label='section_parent'),
        "level": "section_parent",
        "section_label": section_label,
        "paragraf": None, "stk": None, "nr": None,
        "text_plain": text_plain_clean,
        "note_anchors": note_anchors,
        "dom_refs": dom_refs,
        "jv_refs": jv_refs,
        "note_refs": [],
        "note_uuids": [],
        "children": children_ids,
        "anchor_map": anchor_map,
        "created_from": {"child_ids": children_ids},
        # Split-ready struktur (klar til aktivering)
        "part_index": 1,
        "part_total": 1,
        "split_reason": None,
    }


# ------------------------------
# Output helpers
# ------------------------------

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_json(path: str, rows: List[Dict[str, Any]]):
    """Skriv chunks som standard JSON array."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def clean_note_chunk_fields(chunks: List[Dict[str, Any]]) -> None:
    """Fjern overflødige felter fra note-chunks for et renere output."""
    # Note: Med den nye metadata-struktur er der færre redundante felter at fjerne
    # Men vi kan stadig fjerne note_refs og note_uuids da de ikke giver mening for note-chunks
    fields_to_remove = ["note_refs", "note_uuids"]
    for chunk in chunks:
        if chunk.get("level") == "note":
            for field in fields_to_remove:
                if field in chunk:
                    del chunk[field]





def write_csv(path: str, rows: List[Dict[str, Any]]):
    # This function is no longer used, but kept for potential future use.
    # To re-enable, add the call back in main() and ensure pandas is handled.
    pass


# ------------------------------
# CLI
# ------------------------------

def main():
    ap = argparse.ArgumentParser(description='Markdown → juridiske chunks + parent‑noder')
    ap.add_argument('--input', required=True, help='Sti til markdown‑filen')
    ap.add_argument('--out-prefix', default=None, help='Filprefix for output (uden endelse)')
    ap.add_argument('--use-llm-parents', action='store_true', help='Brug LLM til at generere parent chunk bullets')
    ap.add_argument('--fireworks-api-key', help='Fireworks API key (alternativ til miljøvariabel)')
    ap.add_argument('--no-csv', action='store_true', help='Undlad at generere en CSV-fil')
    ap.add_argument('--max-tokens', type=int, default=None, help='Maksimalt antal tokens per chunk (aktiverer auto-split)')
    args = ap.parse_args()

    in_path = args.input
    assert os.path.exists(in_path), f'Inputfil ikke fundet: {in_path}'
    out_prefix = args.out_prefix or os.path.splitext(os.path.basename(in_path))[0].replace(' ', '_')
    
    # Initialize LLM if requested
    llm = None
    if args.use_llm_parents:
        if not FIREWORKS_AVAILABLE:
            safe_print("Warning: Fireworks not available. Install with: pip install fireworks-ai")
            safe_print("Falling back to rule-based parent generation.")
        else:
            safe_print("Initializing Fireworks LLM for parent generation...")
            # Debug: API key provided via arguments (removed for production)
            llm = get_fireworks_llm(api_key=getattr(args, 'fireworks_api_key', None))
            if llm:
                safe_print("LLM initialized successfully!")
            else:
                safe_print("LLM initialization failed. Using rule-based fallback.")

    with open(in_path, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()

    chunks, note_bodies, note_uuid_map = parse_document(lines)
    
    # Build cross-references between chunks and notes
    build_cross_references(chunks, note_uuid_map)
    
    # Clean up orphaned note references
    clean_orphaned_note_references(chunks, note_uuid_map)

    # Apply note context
    apply_note_context(chunks)
    
    # Simple QA check
    safe_print(f"Generated {len(chunks)} chunks with UUID cross-references")

    # Indekser til parent‑bygning
    by_p, by_ps, by_psn, p_intro, sec_to_p = index_by_hierarchy(chunks)

    # 1) stk_parent pr. (p, s) med nr‑børn
    stk_parents: List[Dict[str, Any]] = []
    for (p, s), arr in by_ps.items():
        # tjek om der findes nr‑børn for denne (p,s)
        nr_children = [c for c in chunks if c.get('paragraf') == p and c.get('stk') == s and c.get('level') == 'nr']
        if nr_children:
            parent = make_stk_parent(chunks, p, s, p_intro.get(p), use_llm=args.use_llm_parents, llm=llm)
            if parent:
                stk_parents.append(parent)

    chunks.extend(stk_parents)

    # 2) paragraf_parent pr. § med mindst ét stk (eller stk_parent)
    paragraf_parents: List[Dict[str, Any]] = []
    for p, arr in by_p.items():
        has_stk = any(c.get('paragraf') == p and c.get('level') in ('stk', 'stk_parent') and c.get('nr') is None for c in chunks)
        if has_stk:
            parent = make_paragraf_parent(chunks, p, p_intro.get(p))
            if parent:
                paragraf_parents.append(parent)

    chunks.extend(paragraf_parents)

    # 3) section_parent (kapitel/afsnit)
    section_parents: List[Dict[str, Any]] = []
    for sec_label, paragrafs in sec_to_p.items():
        parent = make_section_parent(sec_label, chunks, paragrafs, p_intro)
        if parent:
            section_parents.append(parent)

    chunks.extend(section_parents)

    # Ryd op i note-felter før output
    clean_note_chunk_fields(chunks)
    
    # Split-logik (klar til aktivering)
    if args.max_tokens:
        safe_print(f"Token-begrænsning aktiveret: {args.max_tokens} tokens per chunk")
        original_count = len(chunks)
        split_chunks = []
        
        for chunk in chunks:
            text = chunk.get("text_plain", "")
            if should_split_chunk(text, args.max_tokens):
                # Split chunk i mindre dele
                parts = split_chunk_by_tokens(chunk, args.max_tokens)
                split_chunks.extend(parts)
            else:
                split_chunks.append(chunk)
        
        chunks = split_chunks
        safe_print(f"Split-resultat: {original_count} -> {len(chunks)} chunks")

    # Skriv filer
    jsonl_path = f"{out_prefix}_chunks.jsonl"
    json_path = f"{out_prefix}_chunks.json"
    write_jsonl(jsonl_path, chunks)
    write_json(json_path, chunks)

    if not args.no_csv:
        csv_path = f"{out_prefix}_chunks.csv"
        # write_csv(csv_path, chunks) # Deaktiveret
        safe_print(f"Wrote {len(chunks)} chunks -> {jsonl_path} + {json_path} (CSV output er deaktiveret)")
    else:
        safe_print(f"Wrote {len(chunks)} chunks -> {jsonl_path} + {json_path}")


if __name__ == '__main__':
    main()
