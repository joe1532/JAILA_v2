#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script til at tilføje markdown-markører til juridiske dokumenter
Tilføjer struktureret markdown baseret på AFSNIT, paragraffer, nummererede punkter og stykker
"""

import re
import argparse
from pathlib import Path
from typing import List, Tuple

class LegalMarkdownProcessor:
    def __init__(self):
        self.patterns = {
            'afsnit': re.compile(r'^AFSNIT\s+([IVX]+[A-Z]*)\.\s*(.+)$', re.MULTILINE),
            'paragraf': re.compile(r'^§\s*(\d+(?:\s*[A-Za-z]+)?)\.\s*(.+)$', re.MULTILINE),
            'nummereret_punkt': re.compile(r'^(\d+)\.\s*(.+)$', re.MULTILINE),
            'nummereret_punkt_parentes': re.compile(r'^(\d+)\)\s*(.+)$', re.MULTILINE),
            'stk': re.compile(r'^(Stk\.\s*\d+)\.\s*(.+)$', re.MULTILINE),
            'lovbekendtgorelse': re.compile(r'^Lovbekendtgørelse\s+(.+)$', re.MULTILINE)
        }
    
    def prevent_markdown_formatting(self, text: str) -> str:
        """
        Forhindrer markdown-parsere i at fortolke juridiske noter som formatering
        ved at indsætte zero-width space mellem parenteser og tal
        
        Args:
            text: Input tekst
            
        Returns:
            Tekst med zero-width space omkring juridiske noter
        """
        # Zero-width space karakter
        zwsp = '\u200B'
        
        # Indsæt zero-width space mellem ( og tal, samt mellem tal og )
        # Dette forhindrer markdown-parsere i at fortolke (tal) som formatering
        modified_text = re.sub(r'\((\d+)\)', f'({zwsp}\\1{zwsp})', text)
        return modified_text
    
    def normalize_whitespace(self, text: str) -> str:
        """Erstatter multiple whitespace-karakterer (inkl. tabs) med et enkelt mellemrum."""
        return re.sub(r'\s+', ' ', text).strip()

    def process_file(self, input_path: Path, output_path: Path) -> bool:
        """Process en fil og tilføj markdown-markører"""
        print(f"Behandler fil: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tjek for YAML frontmatter
        if content.startswith('---'):
            # Split ved første og anden ---
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter = f"---{parts[1]}---\n"
                main_content = parts[2]
            else:
                frontmatter = ""
                main_content = content
        else:
            frontmatter = ""
            main_content = content
        
        # Tilføj markdown-markører
        processed_content = self.add_markdown_markers(main_content)
        
        # Forhindre markdown-parsere i at fortolke juridiske noter som formatering
        final_processed_content = self.prevent_markdown_formatting(processed_content)
        
        # Kombiner frontmatter og processed content
        final_content = frontmatter + final_processed_content
        
        # Gem til output fil
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"Gemt til: {output_path}")
        return True
    
    def add_markdown_markers(self, content: str) -> str:
        """Tilføj markdown-markører til indholdet"""
        lines = content.split('\n')
        processed_lines = []
        
        for line in lines:
            stripped_line = line.strip()
            
            # Skip tomme linjer
            if not stripped_line:
                processed_lines.append(line)
                continue
            
            # Check for AFSNIT (# niveau)
            afsnit_match = self.patterns['afsnit'].match(stripped_line)
            if afsnit_match:
                afsnit_num = afsnit_match.group(1)
                afsnit_title = afsnit_match.group(2)
                processed_lines.append(f"# AFSNIT {afsnit_num}. {afsnit_title}")
                continue
            
            # Check for § paragraffer (## niveau)
            paragraf_match = self.patterns['paragraf'].match(stripped_line)
            if paragraf_match:
                paragraf_num_raw = paragraf_match.group(1)
                paragraf_num = self.normalize_whitespace(paragraf_num_raw) # Normaliser whitespace
                paragraf_title = paragraf_match.group(2)
                processed_lines.append(f"## § {paragraf_num}. {paragraf_title}")
                continue
            
            # Check for Stk. (### niveau)
            stk_match = self.patterns['stk'].match(stripped_line)
            if stk_match:
                stk_num = stk_match.group(1)
                stk_content = stk_match.group(2)
                processed_lines.append(f"### {stk_num}. {stk_content}")
                continue
            
            # Check for nummererede punkter med . (### niveau)
            nummer_match = self.patterns['nummereret_punkt'].match(stripped_line)
            if nummer_match:
                nummer = nummer_match.group(1)
                nummer_content = nummer_match.group(2)
                processed_lines.append(f"### {nummer}. {nummer_content}")
                continue
            
            # Check for nummererede punkter med ) (### niveau)
            nummer_parentes_match = self.patterns['nummereret_punkt_parentes'].match(stripped_line)
            if nummer_parentes_match:
                nummer = nummer_parentes_match.group(1)
                nummer_content = nummer_parentes_match.group(2)
                processed_lines.append(f"### {nummer}) {nummer_content}")
                continue
            
            # Almindelig linje - bevar som den er
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def extract_short_title(self, content: str) -> str:
        """Udtræk en kort titel fra indholdet for overskriften"""
        # Tag de første 50 karakterer og find sidste hele ord
        if len(content) <= 50:
            return content
        
        truncated = content[:50]
        # Find sidste mellemrum for at undgå at afbryde ord
        last_space = truncated.rfind(' ')
        if last_space > 30:  # Kun hvis der er et rimeligt antal karakterer
            return truncated[:last_space] + "..."
        else:
            return truncated + "..."
    
    def clean_content(self, content: str) -> str:
        """Rens indhold for unødvendige mellemrum og linjer"""
        # Fjern multiple tomme linjer
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', content)
        
        # Fjern trailing spaces
        lines = content.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        
        return '\n'.join(cleaned_lines)

def main():
    parser = argparse.ArgumentParser(description='Tilføj markdown-markører til juridiske dokumenter')
    parser.add_argument('--input', '-i', required=True, help='Input fil eller mappe')
    parser.add_argument('--output', '-o', required=True, help='Output fil eller mappe')
    parser.add_argument('--overwrite', action='store_true', help='Overskriv eksisterende filer')
    
    args = parser.parse_args()
    
    processor = LegalMarkdownProcessor()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Behandl enkelt fil
        if output_path.is_dir():
            output_file = output_path / input_path.name  # Same filename, no suffix added
        else:
            output_file = output_path
        
        if output_file.exists() and not args.overwrite:
            print(f"Fil eksisterer allerede: {output_file}")
            print("Brug --overwrite for at overskrive")
            return
        
        processor.process_file(input_path, output_file)
    
    elif input_path.is_dir():
        # Behandl alle .md filer i mappen
        output_path.mkdir(exist_ok=True)
        
        md_files = list(input_path.glob("*.md"))
        if not md_files:
            print("Ingen .md filer fundet i input mappen")
            return
        
        for md_file in md_files:
            output_file = output_path / md_file.name  # Same filename, no suffix added
            
            if output_file.exists() and not args.overwrite:
                print(f"Springer over eksisterende fil: {output_file}")
                continue
            
            processor.process_file(md_file, output_file)
    
    else:
        print(f"Input sti eksisterer ikke: {input_path}")

if __name__ == "__main__":
    main() 