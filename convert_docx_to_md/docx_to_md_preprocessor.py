#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOCX til Markdown Konverter med Indbygget Lovstruktur Preprocessing
Konverterer DOCX filer til en samlet Markdown fil uden markdown-markører
"""

import os
import re
from pathlib import Path
from docx import Document
from typing import Dict, List, Optional, Tuple
import argparse
import logging

class LegalDocxToMarkdownConverter:
    """Konverterer DOCX til Markdown med automatisk lovstruktur preprocessing"""
    
    def __init__(self, input_dir: str = "input", output_dir: str = "output"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.logger = self._setup_logging()
        
        # Opret output mappe
        self.output_dir.mkdir(exist_ok=True)
        
        # Preprocessing regler
        self.preprocessing_rules = {
            'afsnit': r'^(AFSNIT\s+[IVXLCDM]+)\s*\.?\s*(.*)',
            'paragraf': r'^§\s*(\d+[A-Za-z]?)\s*\.?\s*(.*)',
            'stykke': r'^Stk\.\s*(\d+)\s*\.?\s*(.*)',
            'nummer': r'^(\d+)\)\s*(.*)',
            'litra': r'^([a-z])\)\s*(.*)',
            'ophævet': r'^\(Ophævet\)$'
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging konfiguration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def convert_docx_to_markdown(self, docx_path: Path) -> Optional[Path]:
        """
        Konverterer en DOCX fil til Markdown uden markører
        
        Args:
            docx_path: Sti til DOCX filen
            
        Returns:
            Sti til den genererede Markdown fil eller None ved fejl
        """
        try:
            self.logger.info(f"Konverterer: {docx_path}")
            
            # Læs DOCX fil
            doc = Document(docx_path)
            
            # Udtræk paragraphs
            paragraphs = []
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if text:  # Kun ikke-tomme paragraffer
                    paragraphs.append(text)
            
            self.logger.info(f"Udtrukket {len(paragraphs)} paragraffer")
            
            # Foretag preprocessing uden markdown-markører
            processed_paragraphs = []
            for paragraph in paragraphs:
                processed_line = self._preprocess_paragraph_plain(paragraph)
                if processed_line:
                    processed_paragraphs.append(processed_line)
            
            # Byg markdown indhold
            content_lines = []
            
            # Tilføj YAML front matter
            content_lines.append("---")
            content_lines.append(f"source_file: {docx_path.name}")
            content_lines.append(f"converted_at: {self._get_timestamp()}")
            content_lines.append("format: legal_document")
            # Tilføj titel fra filnavn
            title = self._extract_title_from_filename(docx_path.name)
            if title:
                content_lines.append(f"title: {title}")
            content_lines.append("---")
            content_lines.append("")
            
            # Tilføj alle paragraffer
            content_lines.extend(processed_paragraphs)
            
            # Gem som Markdown fil
            output_path = self.output_dir / f"{docx_path.stem}.md"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(content_lines))
            
            self.logger.info(f"Gemt som: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Fejl ved konvertering af {docx_path}: {e}")
            return None
    
    def _preprocess_paragraph_plain(self, paragraph: str) -> str:
        """
        Forbedret preprocessing af juridisk paragraf til ren tekst uden markdown-markører
        
        Args:
            paragraph: Rå paragraf tekst
            
        Returns:
            Preprocessed paragraf uden markdown formatering
        """
        # Spring titler over der allerede er tilføjet
        if paragraph.strip() == "Kildeskatteloven":
            return ""  # Spring duplikeret titel over
        
        # Normale paragraffer - ingen speciel formatering, bare returner som er
        return paragraph
    
    def _extract_title_from_filename(self, filename: str) -> Optional[str]:
        """
        Udtrækker lovtitel fra filnavn
        
        Args:
            filename: Filnavn
            
        Returns:
            Udtrukket titel eller None
        """
        # Fjern fil extension
        stem = Path(filename).stem
        
        # Tjek for standard lovbekendtgørelse format
        # Format: "Lovnavn (dato nr. xxx)"
        title_match = re.match(r'^(.+?)\s*\((.+?)\s+nr\.\s+(.+?)\)$', stem)
        if title_match:
            return title_match.group(1).strip()
        
        # Fallback til hele filnavnet
        return stem
    
    def _get_timestamp(self) -> str:
        """Returnerer ISO timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def convert_all_docx_files(self) -> List[Path]:
        """
        Konverterer alle DOCX filer i input-mappen til separate markdown filer
        
        Returns:
            Liste af stier til genererede Markdown filer
        """
        if not self.input_dir.exists():
            self.logger.error(f"Input mappe findes ikke: {self.input_dir}")
            return []
        
        # Find alle DOCX filer
        docx_files = list(self.input_dir.glob("*.docx"))
        
        # Filtrer temporære filer væk
        docx_files = [f for f in docx_files if not f.name.startswith('~$') and not f.name.startswith('._')]
        
        if not docx_files:
            self.logger.warning("Ingen DOCX filer fundet i input-mappen")
            return []
        
        self.logger.info(f"Fundet {len(docx_files)} DOCX filer til konvertering")
        
        # Konverter hver fil
        converted_files = []
        for docx_file in docx_files:
            try:
                md_path = self.convert_docx_to_markdown(docx_file)
                if md_path:
                    converted_files.append(md_path)
            except Exception as e:
                self.logger.error(f"Fejl ved konvertering af {docx_file}: {e}")
        
        self.logger.info(f"Konverterede {len(converted_files)} filer successfully")
        return converted_files
    
    def validate_conversion(self, md_path: Path) -> Dict:
        """
        Validerer kvaliteten af konverteringen
        
        Args:
            md_path: Sti til Markdown fil
            
        Returns:
            Validerings rapport
        """
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tæl paragraffer og linjer
            lines = content.split('\n')
            non_empty_lines = [line for line in lines if line.strip()]
            
            return {
                "file": str(md_path),
                "total_lines": len(lines),
                "non_empty_lines": len(non_empty_lines),
                "total_characters": len(content),
                "has_yaml_frontmatter": content.startswith("---"),
                "validation_passed": len(non_empty_lines) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Fejl ved validering af {md_path}: {e}")
            return {
                "file": str(md_path),
                "error": str(e),
                "validation_passed": False
            }

def main():
    """CLI interface til konverteren"""
    parser = argparse.ArgumentParser(description="Konverter DOCX til samlet Markdown uden markører")
    parser.add_argument("--input", "-i", default="input", help="Input mappe med DOCX filer")
    parser.add_argument("--output", "-o", default="output", help="Output mappe til Markdown fil")
    parser.add_argument("--validate", "-v", action="store_true", help="Validér konvertering kvalitet")
    parser.add_argument("--verbose", action="store_true", help="Detaljeret logging")
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Opret konverter
    converter = LegalDocxToMarkdownConverter(args.input, args.output)
    
    try:
        # Konverter alle filer til separate filer
        converted_files = converter.convert_all_docx_files()
        
        if converted_files:
            print(f"Konverterede {len(converted_files)} filer successfully:")
            for file_path in converted_files:
                print(f"  - {file_path.name}")
            
            if args.validate:
                print("\nValidering af konverterede filer:")
                for md_path in converted_files:
                    validation = converter.validate_conversion(md_path)
                    print(f"  {md_path.name}: {validation.get('non_empty_lines', 0)} linjer, {validation.get('total_characters', 0)} karakterer")
        else:
            print("Ingen filer blev konverteret")
    
    except KeyboardInterrupt:
        print("\nAfbrudt af bruger")
    except Exception as e:
        print(f"Fejl: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 