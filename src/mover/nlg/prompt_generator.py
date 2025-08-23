import json
import itertools
import random
import copy
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import OrderedDict
from pyrealb import loadEn
import re
from mover.nlg.sentence_generation.vocab import Vocab
from mover.nlg.data_classes import Motion, Object
from mover.nlg.sentence_generation.sentence_composer import SentenceComposer


class PromptGenerator:
    """Main logic for generating natural language descriptions of motions from motion scene graphs."""
    
    def __init__(self, vocab: Optional[Vocab] = None, sampling_config: Optional[Dict] = None):
        
        ## Default sampling configuration
        self.sampling_config = sampling_config or {
            "enabled": True,
            "max_per_group": 2,  ## Maximum sentences per (template, unit) group
            "multi_motion_downsample_num": 50,
        }
        
        self.vocab = vocab or Vocab()
        self.composer = SentenceComposer(self._load_patterns(), self.vocab, self.sampling_config)
        loadEn()
    
    
    def _load_patterns(self):
        """Load sentence patterns from assets."""
        assets_dir = Path(__file__).parent / "assets"
        patterns_file = assets_dir / "sentence_patterns.json"
        
        if patterns_file.exists():
            with open(patterns_file, 'r') as f:
                return json.load(f)
        else:
            raise FileNotFoundError(f"Patterns file not found: {patterns_file}")
    
    
    def generate(self, gen_data: Dict[str, Any]) -> List[Dict]:
        """
        Main entry point for generating prompts from motion data.
        
        Args:
            gen_data: Dictionary containing:
                - "motions": List of Motion objects
                - "relations": List of temporal relations (optional)
        
        Returns:
            List of dictionaries with sentence metadata:
                - "sentence": Generated sentence string
                - "structure": List of template patterns used
                - "ordering": List of modifier orderings  
                - "unseen synonyms": Dict of detected synonyms
        """
        motions = gen_data["motions"]
        relations = gen_data.get("relations", [None] * len(motions))
        
        if len(motions) == 1:
            return self._generate_single_motion_sentences(motions[0])
        else:
            return self._generate_multi_motion_sentences(motions, relations)
    
    
    def _generate_single_motion_sentences(self, motion: Motion) -> List[Dict]:
        """Generate sentences for a single motion using all available patterns."""
        all_results = []
        
        ## Available sentence patterns for single motion
        patterns = ["present", "present_pass", "imperative", "imperative_ccomp", "imperative_advcl"]
        
        print(f"Generating sentences for {motion.type} motion")
        for pattern in patterns:
            print(f"    Processing pattern: {pattern}")
            sentences, labels = self.composer.compose(motion, pattern)
            ## Realize sentences to strings
            sentences = [s.realize() for s in sentences]
            
            ## Create metadata for each generated sentence
            for i, sentence in enumerate(sentences):
                unseen_synonyms = self.vocab.detect_synonyms_in_sentence(sentence)
                
                all_results.append({
                    "sentence": sentence,
                    "structure": [pattern],
                    "ordering": labels[i][:-1] if i < len(labels) else [],
                    "unseen synonyms": unseen_synonyms
                })
        
        return self._remove_duplicates_with_metadata(all_results)
    
    
    def _generate_multi_motion_sentences(self, motions: List[Motion], relations: List[str]) -> List[Dict]:
        """Generate sentences for exactly two motions with temporal relations."""
        if len(motions) != 2:
            ## For now, only handle exactly 2 motions
            return self._generate_single_motion_sentences(motions[0])
        
        print(f"Generating sentences for {len(motions)} motions")
        motion1, motion2 = motions[0], motions[1]
        relation = relations[1] if len(relations) > 1 else None
                
        ## Generate sentences for each motion individually
        motion1_sentences = self._generate_single_motion_sentences(motion1)
        motion2_sentences = self._generate_single_motion_sentences(motion2)
        
        ## Limit sentences for performance (sample from each)
        motion1_sentences = random.sample(motion1_sentences, min(self.sampling_config["multi_motion_downsample_num"], len(motion1_sentences)))
        motion2_sentences = random.sample(motion2_sentences, min(self.sampling_config["multi_motion_downsample_num"], len(motion2_sentences)))
        
        ## Determine object mode for second motion
        obj_mode = self._get_object_mode(motion1, motion2)
        
        ## Regenerate second motion sentences with appropriate object mode if needed
        if obj_mode != "default":
            motion2_sentences = []
            ## Get patterns that support concatenation from config
            patterns_config = self._load_patterns()
            concat_rules = patterns_config.get("concat_rules", {})
            compatible_patterns = []
            for pattern, allowed_next in concat_rules.items():
                if allowed_next is not None:  ## null means no concatenation allowed
                    compatible_patterns.append(pattern)
            
            ## Use compatible patterns or fallback to basic ones
            patterns = compatible_patterns if compatible_patterns else ["present"]
            
            for pattern in patterns:
                sentences, labels = self.composer.compose(motion2, pattern, obj_mode=obj_mode)
                sentences = [s.realize() for s in sentences]
                
                for i, sentence in enumerate(sentences):
                    unseen_synonyms = self.vocab.detect_synonyms_in_sentence(sentence)
                    motion2_sentences.append({
                        "sentence": sentence,
                        "structure": [pattern],
                        "ordering": labels[i][:-1] if i < len(labels) else [],
                        "unseen synonyms": unseen_synonyms
                    })
            
            motion2_sentences = random.sample(motion2_sentences, min(self.sampling_config["multi_motion_downsample_num"], len(motion2_sentences)))
        
        ## Combine sentences based on temporal relation
        return self._combine_motion_sentences(motion1_sentences, motion2_sentences, relation)


    def _get_object_mode(self, motion1: Motion, motion2: Motion) -> str:
        """Determine object reference mode for second motion."""
        if motion1.agent == motion2.agent:
            ## Same agent - can use pronoun or omit
            return "pron"  ## Use pronoun for clarity
        else:
            ## Different agents - use full description
            return "default"


    def _combine_motion_sentences(self, motion1_sentences: List[Dict], motion2_sentences: List[Dict], relation: str) -> List[Dict]:
        """Combine two motion sentences with temporal connectors."""
        print(f"Combining {len(motion1_sentences)} and {len(motion2_sentences)} sentences")
        combined_results = []
        
        ## Get temporal connectors from patterns
        connectors = self._get_temporal_connectors(relation)
        
        ## Get concat rules to determine which patterns can be concatenated
        patterns_config = self._load_patterns()
        concat_rules = patterns_config.get("concat_rules", {})
        
        ## Limit combinations for performance
        max_combinations = self.sampling_config["multi_motion_downsample_num"] ** 2
        total_possible = len(motion1_sentences) * len(motion2_sentences) * len(connectors["standalone"] + connectors["concat"])
        
        if total_possible > max_combinations:
            ## Sample combinations
            combinations = []
            for _ in range(max_combinations):
                s1 = random.choice(motion1_sentences)
                s2 = random.choice(motion2_sentences)
                ## Check if concatenation is allowed for this pattern combination
                allowed_connection_types = self._get_allowed_connection_types(s1, s2, concat_rules)
                if allowed_connection_types:
                    conn_type = random.choice(allowed_connection_types)
                    conn = random.choice(connectors[conn_type])
                    combinations.append((s1, s2, conn, conn_type))
        else:
            ## Generate all combinations
            combinations = []
            for s1 in motion1_sentences:
                for s2 in motion2_sentences:
                    allowed_connection_types = self._get_allowed_connection_types(s1, s2, concat_rules)
                    for conn_type in allowed_connection_types:
                        for conn in connectors[conn_type]:
                            combinations.append((s1, s2, conn, conn_type))
        
        ## Create combined sentences
        for s1, s2, connector, conn_type in combinations:
            combined_sentence = self._join_sentences(s1["sentence"], s2["sentence"], connector, conn_type)
            
            combined_results.append({
                "sentence": combined_sentence,
                "structure": s1["structure"] + s2["structure"],
                "ordering": s1["ordering"] + s2["ordering"],
                "unseen synonyms": {**s1["unseen synonyms"], **s2["unseen synonyms"]},
                "temporal_relation": relation,
                "connection_type": conn_type
            })
        
        return self._remove_duplicates_with_metadata(combined_results)


    def _get_allowed_connection_types(self, s1: Dict, s2: Dict, concat_rules: Dict) -> List[str]:
        """Determine which connection types are allowed based on concat rules."""
        ## Always allow standalone connections
        allowed_types = ["standalone"]
        
        ## Check if concatenation is allowed
        s1_pattern = s1["structure"][0] if s1["structure"] else None
        s2_pattern = s2["structure"][0] if s2["structure"] else None
        
        if s1_pattern in concat_rules:
            allowed_next_patterns = concat_rules[s1_pattern]
            if allowed_next_patterns is not None and s2_pattern in allowed_next_patterns:
                allowed_types.append("concat")
        
        return allowed_types


    def _get_temporal_connectors(self, relation: str) -> Dict[str, List[str]]:
        """Get temporal connectors for a relation."""
        if not relation:
            return {"standalone": [""], "concat": [""]}
        
        ## Load from patterns file
        patterns = self._load_patterns()
        temporal_connectors = patterns.get("temporal_connectors", {})
        
        ## Map relation names to pattern keys
        relation_mapping = {
            "precedes": "before",
            "preceded_by": "after", 
            "overlaps": "overlaps"
        }
        
        pattern_key = relation_mapping.get(relation, relation)
        
        if pattern_key in temporal_connectors:
            return temporal_connectors[pattern_key]
        
        ## If not found, return empty connectors (no fallback)
        return {"standalone": [""], "concat": [""]}


    def _join_sentences(self, sentence1: str, sentence2: str, connector: str, conn_type: str) -> str:
        """Join two sentences with a temporal connector."""
        if not connector:
            return f"{sentence1} {sentence2}"
        
        ## Clean up sentences first
        sentence1 = sentence1.strip()
        sentence2 = sentence2.strip()
        
        if conn_type == "standalone":
            ## Separate sentences: "Sentence1. Connector, sentence2."
            ## Ensure first sentence ends with period
            if not sentence1.endswith('.'):
                sentence1 += '.'
            ## Make second sentence lowercase
            sentence2_lower = sentence2[0].lower() + sentence2[1:] if len(sentence2) > 1 else sentence2.lower()
            return f"{sentence1} {connector.capitalize()}, {sentence2_lower}"
        
        elif conn_type == "concat":
            ## Concatenated: "Sentence1, connector sentence2."
            ## Remove period from first sentence and lowercase second
            sentence1_clean = sentence1.rstrip('.')
            sentence2_lower = sentence2[0].lower() + sentence2[1:] if len(sentence2) > 1 else sentence2.lower()
            return f"{sentence1_clean}, {connector} {sentence2_lower}"
        
        return f"{sentence1} {sentence2}"
    
    
    def _remove_duplicates_with_metadata(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate sentences while preserving metadata."""
        seen_sentences = OrderedDict()
        
        for result in results:
            sentence = result["sentence"]
            
            ## Clean up excessive commas and spacing issues
            sentence = self._clean_excessive_commas(sentence)
            result["sentence"] = sentence
            
            if sentence not in seen_sentences:
                seen_sentences[sentence] = result
        
        return list(seen_sentences.values())
    
    
    def _clean_excessive_commas(self, sentence: str) -> str:
        """Clean up excessive commas and spacing issues in sentences."""
        if not sentence:
            return sentence
        
        ## Replace multiple consecutive commas with single comma
        sentence = re.sub(r',\s*,(\s*,)*', ',', sentence)
        sentence = re.sub(r',\s*\.', '.', sentence)
        sentence = re.sub(r'\s+,', ',', sentence)
        sentence = re.sub(r',\s+', ', ', sentence)
        sentence = re.sub(r'\s+', ' ', sentence)
        sentence = re.sub(r'\s+\.', '.', sentence)
        
        return sentence.strip()
    
    
    def get_available_synonyms(self) -> List[Tuple]:
        """Get all available LLM-produced synonyms for debugging/testing."""
        return self.vocab.get_llm_produced_synonyms()
