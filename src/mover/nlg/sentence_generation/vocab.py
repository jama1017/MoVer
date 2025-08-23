"""
Unified vocabulary management for motion NLG.
"""
import json
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from pyrealb import Adv, PP, V, N, A, D, P, NP, C, CP, VP, loadEn, addToLexicon

class LazyPyRealB:
    """Unified lazy wrapper for all PyRealB objects - defers creation until realization."""
    
    def __init__(self, constructor_type=None, *args, vocab_data=None, vocab_instance=None, **kwargs):
        """
        Create a lazy PyRealB object.
        
        Args:
            constructor_type: PyRealB class name (e.g., "PP", "NP", "V", "N") or None for vocab-based
            *args: Arguments to pass to the PyRealB constructor
            vocab_data: Raw vocabulary data dict (for vocab-based objects)
            vocab_instance: Vocab instance (for vocab-based objects)
            **kwargs: Keyword arguments to pass to the PyRealB constructor
        """
        self.constructor_type = constructor_type
        self.args = args
        self.kwargs = kwargs
        self.vocab_data = vocab_data
        self.vocab_instance = vocab_instance
        self._pyrealb_obj = None
    
    @classmethod
    def from_vocab(cls, vocab_data: Dict[str, Any], vocab_instance: 'Vocab'):
        """Create a lazy object from vocabulary data."""
        return cls(vocab_data=vocab_data, vocab_instance=vocab_instance)
    
    @classmethod
    def phrase(cls, phrase_type: str, *args, **kwargs):
        """Create a lazy phrase (PP, NP, AdvP, etc.)."""
        return cls(phrase_type, *args, **kwargs)
    
    def get_pyrealb(self):
        """Create the actual PyRealB object when needed."""
        if self._pyrealb_obj is None:
            if self.vocab_data is not None and self.vocab_instance is not None:
                # Vocab-based object
                self._pyrealb_obj = self.vocab_instance._to_pyrealb(self.vocab_data)
            elif self.constructor_type is not None:
                # Direct PyRealB construction
                # Recursively resolve any lazy args
                resolved_args = []
                for arg in self.args:
                    if isinstance(arg, LazyPyRealB):
                        resolved_args.append(arg.get_pyrealb())
                    elif isinstance(arg, list):
                        resolved_args.append([item.get_pyrealb() if isinstance(item, LazyPyRealB) else item for item in arg])
                    elif arg is None:
                        resolved_args.append(arg)
                    else:
                        resolved_args.append(arg)
                
                # Import PyRealB classes dynamically
                from pyrealb import PP, NP, AdvP, SP, VP, S, CP, V, N, A, D, Adv, P, C
                phrase_classes = {
                    "PP": PP, "NP": NP, "AdvP": AdvP, "SP": SP, "VP": VP, "S": S, "CP": CP,
                    "V": V, "N": N, "A": A, "D": D, "Adv": Adv, "P": P, "C": C
                }
                
                if self.constructor_type in phrase_classes:
                    self._pyrealb_obj = phrase_classes[self.constructor_type](*resolved_args, **self.kwargs)
                else:
                    # Fallback for unknown types
                    self._pyrealb_obj = resolved_args[0] if resolved_args else None
            else:
                # No data to construct from
                self._pyrealb_obj = None
        
        return self._pyrealb_obj
    
    def realize(self):
        """Realize to text."""
        pyrealb_obj = self.get_pyrealb()
        if pyrealb_obj is None:
            return ""
        return pyrealb_obj.realize() if hasattr(pyrealb_obj, 'realize') else str(pyrealb_obj)
    
    def __str__(self):
        return self.realize()
    
    def __repr__(self):
        if self.vocab_data:
            return f"LazyPyRealB.from_vocab({self.vocab_data})"
        else:
            return f"LazyPyRealB.phrase('{self.constructor_type}', {self.args}, {self.kwargs})"


class VocabList:
    
    def __init__(self, vocab_items: List[Dict], vocab_instance: 'Vocab'):
        self.vocab_items = vocab_items
        self.vocab_instance = vocab_instance
    
    def __len__(self):
        return len(self.vocab_items)
    
    def __iter__(self):
        return iter(self.vocab_items)
    
    def to_pyrealb_list(self):
        """Convert all items to PyRealB objects."""
        return [self.vocab_instance._to_pyrealb(item) for item in self.vocab_items]


class Vocab:
    """Unified vocabulary for motion descriptions."""
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            # Default to assets directory relative to this file
            current_dir = Path(__file__).parent.parent  # This file is in mover_nlg/nlg/
            data_dir = current_dir / "assets"     # So assets is in mover_nlg/nlg/assets/
        self.data_dir = Path(data_dir)
        self.vocab_data = self._load_vocab_data()
        self._setup_pyrealb_lexicon()
    
    
    def _load_vocab_data(self) -> Dict[str, Any]:
        """Load vocabulary data from JSON file."""
        json_path = self.data_dir / "vocab.json"
        if json_path.exists():
            with open(json_path, 'r') as f:
                return json.load(f)
        return {}
    

    def _setup_pyrealb_lexicon(self):
        """Add custom words to PyRealB lexicon."""
        loadEn()
        
        lexicon_additions = self.vocab_data.get("lexicon_additions", {})
        
        # Add adjectives
        for adj_data in lexicon_additions.get("adjectives", []):
            addToLexicon(adj_data["word"], {"A": {"tab": adj_data["tab"]}})
        
        # Add verbs  
        for verb_data in lexicon_additions.get("verbs", []):
            addToLexicon(verb_data["word"], {"V": {"tab": verb_data["tab"]}})
        
        # Add nouns
        for noun_data in lexicon_additions.get("nouns", []):
            addToLexicon(noun_data["word"], {"N": {"tab": noun_data["tab"]}})
        
        # Add adverbs
        for adv_data in lexicon_additions.get("adverbs", []):
            addToLexicon(adv_data["word"], {"Adv": {"tab": adv_data["tab"]}})
    

    def _to_pyrealb(self, item_def: Dict[str, Any]):
        """Convert JSON definition to PyRealB object."""
        if item_def["type"] == "V":
            return V(item_def["word"])
        elif item_def["type"] == "N":
            noun = N(item_def["word"])
            ## Handle plural nouns
            if item_def.get("plural", False):
                noun = noun.n("p")
            return noun
        elif item_def["type"] == "A":
            adj = A(item_def["word"])
            if "comp" in item_def:
                adj = adj.f(item_def["comp"])
            return adj
        elif item_def["type"] == "Adv":
            return Adv(item_def["word"])
        elif item_def["type"] == "D":
            return D(item_def["word"])
        elif item_def["type"] == "P":
            return P(item_def["word"])
        elif item_def["type"] == "C":
            return C(item_def["word"])
        elif item_def["type"] == "PP":
            prep = P(item_def["prep"])
            if "np" in item_def:
                np = self._build_np(item_def["np"])
                return PP(prep, np)
            elif "noun" in item_def:
                noun = N(item_def["noun"])
                ## Handle plural nouns in PP
                if item_def.get("plural", False):
                    noun = noun.n("p")
                return PP(prep, noun)
            elif "adv" in item_def:
                return PP(prep, Adv(item_def["adv"]))
            else:
                return PP(prep)
        elif item_def["type"] == "VP":
            ## Handle Verb Phrases
            parts = []
            if "verb" in item_def:
                verb = V(item_def["verb"]["word"])
                if "tense" in item_def["verb"]:
                    verb = verb.t(item_def["verb"]["tense"])
                parts.append(verb)
            if "adj" in item_def:
                parts.append(A(item_def["adj"]))
            if "prep" in item_def:
                parts.append(P(item_def["prep"]))
            if "pp" in item_def:
                ## The pp field is a direct NP structure, not a typed object
                pp = self._build_np(item_def["pp"])
                parts.append(pp)
            return VP(*parts)
        else:
            return item_def.get("word", str(item_def))  # fallback to string
    

    def _build_np(self, np_def: Dict[str, Any]):
        """Build NP from definition."""
        parts = []
        if "det" in np_def:
            parts.append(D(np_def["det"]))
        if "adj" in np_def:
            if isinstance(np_def["adj"], dict):
                adj = A(np_def["adj"]["word"])
                if "comp" in np_def["adj"]:
                    adj = adj.f(np_def["adj"]["comp"])
                parts.append(adj)
            else:
                parts.append(A(np_def["adj"]))
        if "adv" in np_def:
            parts.append(Adv(np_def["adv"]))
        if "verb" in np_def:
            verb = V(np_def["verb"]["word"])
            if "tense" in np_def["verb"]:
                verb = verb.t(np_def["verb"]["tense"])
            parts.append(verb)
        if "noun" in np_def:
            noun = N(np_def["noun"])
            ## Handle plural nouns
            if np_def.get("plural", False):
                noun = noun.n("p")
            parts.append(noun)
        if "noun2" in np_def:
            noun2 = N(np_def["noun2"])
            ## Handle plural nouns for noun2 as well
            if np_def.get("plural2", False):
                noun2 = noun2.n("p")
            parts.append(noun2)
        
        ## Build the NP
        np = NP(*parts)
        
        ## Handle prep2 (like "of" in "the right of")
        if "prep2" in np_def:
            prep2 = P(np_def["prep2"])
            return PP(prep2, np)
        
        return np
    

    def get_vocab(self, section: str, subsection: str, phrase_type: str) -> 'VocabList':
        """Vocabulary getter that returns a simple vocab list."""
        # Get raw vocab data
        section_data = self.vocab_data.get(section, {})
        if subsection not in section_data:
            return VocabList([], self)
        
        subsection_data = section_data[subsection]
        
        # Handle direct arrays (like determiners, location_nouns, etc.)
        if isinstance(subsection_data, list):
            return VocabList(subsection_data, self)
        
        # Handle nested dictionaries with phrase_type
        if isinstance(subsection_data, dict):
            vocab_items = subsection_data.get(phrase_type, [])
            if not vocab_items:
                return VocabList([], self)
            return VocabList(vocab_items, self)
        
        # Return empty if neither list nor dict
        return VocabList([], self)
    

    def to_pyrealb_objects(self, vocab_items: List[Dict]) -> List[Any]:
        """Convert raw vocab items to PyRealB objects when needed."""
        if not vocab_items:
            return []
        return [self._to_pyrealb(item) for item in vocab_items]


    def get_all_keys(self, section: str) -> List[str]:
        """Get all available keys for a section."""
        return list(self.vocab_data.get(section, {}).keys())
    

    def get_vocab_section(self, section: str, subsection: Optional[str] = None, 
                         phrase_type: Optional[str] = None) -> List[Any]:
        """Get vocabulary from any section with caching."""
        # Use the main get_vocab method if we have all parameters
        if subsection and phrase_type:
            return self.get_vocab(section, subsection, phrase_type)
        
        # For partial lookups, don't cache (less common case)
        section_data = self.vocab_data.get(section, {})
        
        if subsection:
            subsection_data = section_data.get(subsection, {})
            # Handle direct arrays
            if isinstance(subsection_data, list):
                return [self._to_pyrealb(item) for item in subsection_data]
            section_data = subsection_data
        
        if phrase_type:
            vocab_items = section_data.get(phrase_type, [])
        else:
            vocab_items = section_data
        
        if isinstance(vocab_items, list):
            return [self._to_pyrealb(item) for item in vocab_items]
        
        return vocab_items
    
    
    ## Synonym tracking functionality (based on gen_data.py)
    def get_llm_produced_synonyms(self):
        """
        Extract LLM-produced synonyms from vocabulary labels.
        Returns list of (synonym_forms, category) tuples.
        Based on the original get_llm_produced_synonyms function from gen_data.py.
        """
        synonym_list = []
        
        if 'synonym_labels' not in self.vocab_data:
            return synonym_list
            
        synonym_labels = self.vocab_data['synonym_labels']
        
        ## Process directions
        if 'directions' in synonym_labels:
            direction_labels = synonym_labels['directions'] 
            direction_vocab = self.vocab_data.get('directions', {})
            
            for direction_key in direction_vocab.keys():
                if direction_key in direction_labels:
                    ## Process ADV and PHRASE for each direction
                    for phrase_type in direction_vocab[direction_key]:
                        vocab_items = direction_vocab[direction_key][phrase_type]
                        if phrase_type in direction_labels[direction_key]:
                            labels = direction_labels[direction_key][phrase_type]
                            
                            for j, vocab_item in enumerate(vocab_items):
                                if j < len(labels) and labels[j] == 1:
                                    ## Extract word/phrase text
                                    phrase_text = self._extract_vocab_text(vocab_item)
                                    if phrase_text:
                                        synonym_list.append(([phrase_text], "direction"))
        
        ## Process motions  
        if 'motions' in synonym_labels:
            motion_labels = synonym_labels['motions']
            motion_vocab = self.vocab_data.get('motions', {})
            
            for motion_key in motion_vocab.keys():
                if motion_key in motion_labels:
                    ## Process each verb type (VERB, VERB_UP, etc.)
                    for verb_type in motion_vocab[motion_key]:
                        if verb_type in motion_labels[motion_key]:
                            vocab_items = motion_vocab[motion_key][verb_type]
                            labels = motion_labels[motion_key][verb_type]
                            
                            for j, vocab_item in enumerate(vocab_items):
                                if j < len(labels) and labels[j] == 1:
                                    word = self._extract_vocab_text(vocab_item)
                                    if word:
                                        ## For verbs, generate multiple forms like original
                                        if verb_type.startswith('VERB'):
                                            try:
                                                from pyrealb import V
                                                verb_forms = [
                                                    V(word).t("b").realize(),
                                                    V(word).t("pp").realize(), 
                                                    V(word).n("s").realize()
                                                ]
                                                synonym_list.append((verb_forms, "motion"))
                                            except:
                                                ## Fallback if pyrealb fails
                                                synonym_list.append(([word], "motion"))
                                        else:
                                            synonym_list.append(([word], "motion"))
        
        return synonym_list
    
    
    def _extract_vocab_text(self, vocab_item):
        """Extract text from vocabulary item (handles various formats)."""
        if isinstance(vocab_item, dict):
            if 'word' in vocab_item:
                return vocab_item['word']
            elif 'text' in vocab_item:
                return vocab_item['text']
            ## Handle complex phrase structures
            elif 'prep' in vocab_item:
                prep = vocab_item['prep']
                if 'np' in vocab_item and isinstance(vocab_item['np'], dict):
                    np_dict = vocab_item['np']
                    parts = [prep]
                    if 'det' in np_dict:
                        parts.append(str(np_dict['det']))
                    if 'adj' in np_dict:
                        adj_val = np_dict['adj']
                        if isinstance(adj_val, dict) and 'word' in adj_val:
                            parts.append(adj_val['word'])
                        else:
                            parts.append(str(adj_val))
                    if 'noun' in np_dict:
                        parts.append(str(np_dict['noun']))
                    if 'noun2' in np_dict:
                        parts.append(str(np_dict['noun2']))
                    return ' '.join(parts)
                elif 'noun' in vocab_item:
                    return f"{prep} {vocab_item['noun']}"
                return prep
        elif isinstance(vocab_item, str):
            return vocab_item
        return ''
    
    
    def detect_synonyms_in_sentence(self, sentence: str):
        """
        Detect unseen synonyms in a sentence.
        Returns dict with synonym categories as keys and found synonyms as values.
        Only includes the exact form found in the sentence.
        """
        synonyms_found = {}
        synonym_list = self.get_llm_produced_synonyms()
        
        # Convert sentence to lowercase for case-insensitive matching
        sentence_lower = sentence.lower()
        sentence_words = sentence_lower.split()
        
        for synonym_forms, category in synonym_list:
            for synonym in synonym_forms:
                if synonym and synonym.lower() in sentence_lower:
                    # Find the exact form used in the sentence
                    synonym_lower = synonym.lower()
                    
                    # For single words, find the exact case used
                    if ' ' not in synonym_lower:
                        for word in sentence.split():
                            if word.lower().rstrip('.,!?;:') == synonym_lower:
                                if category not in synonyms_found:
                                    synonyms_found[category] = []
                                actual_word = word.rstrip('.,!?;:')  # Remove punctuation
                                if actual_word not in synonyms_found[category]:
                                    synonyms_found[category].append(actual_word)
                                break
                    else:
                        # For phrases, find the exact case used
                        start_idx = sentence_lower.find(synonym_lower)
                        if start_idx != -1:
                            actual_phrase = sentence[start_idx:start_idx + len(synonym)]
                            if category not in synonyms_found:
                                synonyms_found[category] = []
                            if actual_phrase not in synonyms_found[category]:
                                synonyms_found[category].append(actual_phrase)
        
        return synonyms_found