import math
import random
import itertools
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from itertools import product
from pyrealb import S, VP, V, NP, SP, PhraseEn, AdvP, loadEn
from mover.nlg.utils import divide_list_into_three
from mover.nlg.sentence_generation.vocab import Vocab, LazyPyRealB
from mover.nlg.data_classes import Motion
from mover.nlg.sentence_generation.sentence_unit import SentenceUnit
    

class SentenceComposer:
    """
    Compose sentences from sentence units with grouped sampling.
    
    Groups results by (template, unit_idx, sequence) and samples from each group
    to ensure balanced representation across different sentence structures.
    """
    
    def __init__(self, patterns: Dict, vocab: Vocab, sampling_config: Dict):
        self.patterns = patterns
        self.vocab = vocab
        self.sampling_config = sampling_config
        loadEn()
        
        ## Initialize the template functions
        self.template_functions = {
            "present": self._present_template,
            "present_pass": self._present_pass_template,
            "imperative": self._imperative_template,
            "imperative_ccomp": self._imperative_ccomp_template,
            "imperative_advcl": self._imperative_advcl_template,
        }


    def _resolve_if_lazy(self, obj):
        """Resolve object if it's a LazyPyRealB, otherwise return as-is."""
        return obj.get_pyrealb() if isinstance(obj, LazyPyRealB) else obj


    def _present_template(self, obj_cont, verb_cont, prefix_list, infix_list, suffix_list) -> Tuple[S, List[str]]:
        """Present tense template - extracted from present_f()."""
        prefix, infix, suffix = prefix_list[0], infix_list[0], suffix_list[0]
        
        # Resolve lazy objects only when needed for PyRealB construction
        prefix = self._resolve_if_lazy(prefix)
        infix = self._resolve_if_lazy(infix)
        suffix = self._resolve_if_lazy(suffix)
            
        if infix:
            obj_cont = NP(obj_cont).a(',')
            infix = SP(infix).a(',')
        prefix_names, infix_names, suffix_names = prefix_list[1], infix_list[1], suffix_list[1]
        sequence = prefix_names + ["object"] + infix_names + ["verb"] + suffix_names + ["present"]
        return S(prefix, obj_cont, infix, VP(verb_cont.t("p")), suffix), sequence
    
    
    def _present_pass_template(self, obj_cont, verb_cont, prefix_list, infix_list, suffix_list) -> Tuple[S, List[str]]:
        """Present passive template - extracted from present_pass_f()."""
        prefix, infix, suffix = prefix_list[0], infix_list[0], suffix_list[0]
        
        # Resolve lazy objects only when needed for PyRealB construction
        prefix = self._resolve_if_lazy(prefix)
        infix = self._resolve_if_lazy(infix)
        suffix = self._resolve_if_lazy(suffix)
        
        if infix:
            obj_cont = NP(obj_cont).a(',')
            infix = SP(infix).a(',')
        prefix_names, infix_names, suffix_names = prefix_list[1], infix_list[1], suffix_list[1]
        sequence = prefix_names + ["object"] + infix_names + ["verb"] + suffix_names + ["present pass"]
        return S(prefix, obj_cont, infix, VP(V("be"), verb_cont.t("pp")), suffix), sequence
    

    def _imperative_template(self, obj_cont, verb_cont, prefix_list, infix_list, suffix_list) -> Tuple[S, List[str]]:
        """Imperative template - extracted from imperative_f()."""
        prefix, infix, suffix = prefix_list[0], infix_list[0], suffix_list[0]
        
        # Resolve lazy objects only when needed for PyRealB construction
        prefix = self._resolve_if_lazy(prefix)
        infix = self._resolve_if_lazy(infix)
        suffix = self._resolve_if_lazy(suffix)
        
        prefix_names, infix_names, suffix_names = prefix_list[1], infix_list[1], suffix_list[1]
        sequence = prefix_names + ["verb"] + ["object"] + infix_names + suffix_names + ["imperative"]
        return S(prefix, VP(verb_cont.t('ip')), obj_cont, infix, suffix), sequence
    

    def _imperative_ccomp_template(self, obj_cont, verb_cont, prefix_list, infix_list, suffix_list) -> Tuple[S, List[str]]:
        """Imperative with ccomp template - extracted from imperative_ccomp_f()."""
        prefix, infix, suffix = prefix_list[0], infix_list[0], suffix_list[0]
        
        # Resolve lazy objects only when needed for PyRealB construction
        prefix = self._resolve_if_lazy(prefix)
        infix = self._resolve_if_lazy(infix)
        suffix = self._resolve_if_lazy(suffix)
        
        prefix_names, infix_names, suffix_names = prefix_list[1], infix_list[1], suffix_list[1]
        sequence = prefix_names + ["object"] + ["verb"] + infix_names + suffix_names + ["imperative_ccomp"]
        return S(prefix, VP(V("make").t('ip'), obj_cont, verb_cont.t('ip')), infix, suffix), sequence
    

    def _imperative_advcl_template(self, obj_cont, verb_cont, prefix_list, infix_list, suffix_list) -> Tuple[S, List[str]]:
        """Imperative with advcl template - extracted from imperative_advcl_f()."""
        prefix, infix, suffix = prefix_list[0], infix_list[0], suffix_list[0]
        
        # Resolve lazy objects only when needed for PyRealB construction
        prefix = self._resolve_if_lazy(prefix)
        infix = self._resolve_if_lazy(infix)
        suffix = self._resolve_if_lazy(suffix)
        
        prefix_names, infix_names, suffix_names = prefix_list[1], infix_list[1], suffix_list[1]
        sequence = prefix_names + ["object"] + ["verb"] + infix_names + suffix_names + ["imperative_advcl"]
        return S(prefix, VP(V("animate").t('ip'), obj_cont, verb_cont.t("b-to")), infix, suffix), sequence
    

    def _format_object_content(self, obj_cont: Any, obj_mode: str) -> Any:
        """Format object content - extracted from format_obj_cont()."""
        match obj_mode:
            case "default":
                return obj_cont
            case "pron":
                return NP(obj_cont).pro()
            case "omit":
                return None
        return obj_cont
    

    def _generate_unit_combinations(self, base_units: Dict) -> List[Dict]:
        """Generate all unit combinations with modifier permutations."""
        units = []
        
        # Get modifier lists
        direction = self.resolve_modifier_content(base_units["dir"])
        magnitude = self.resolve_modifier_content(base_units["mag"])
        origin = base_units["orig"]
        duration = base_units["dur"]
        post = base_units["post"]
        
        # All permutations of 5 modifiers
        ids = [0, 1, 2, 3, 4]
        names = ["direction", "magnitude", "origin", "duration", "post"]
        items = [direction, magnitude, origin, duration, post]
        
        mod_id_permutations = list(itertools.permutations(ids))
        for mod_id_perm in mod_id_permutations:
            mod_perm = [items[i] for i in mod_id_perm]
            sequence = [names[i] for i in mod_id_perm]
            units.append({
                "OBJ": base_units["obj"],
                "VERB": base_units["verb"],
                "MOD": mod_perm,
                "SEQUENCE": sequence
            })
        
        return units
    
    
    def assemble_units(self, units: List[Dict], template_name: str, **kwargs) -> Tuple[List[S], List[List[str]]]:
        """
        Generate sentences with all modifier positions, grouped by template and unit ordering.
        
        Results are grouped by (template_name, unit_idx, sequence) and sampled to ensure
        balanced representation across different sentence structures.
        """
        obj_mode = kwargs.get("obj_mode", "default")
        
        template_f = self.template_functions.get(template_name, self._present_template)
        
        grouped_results = {}
        for unit_idx, unit in enumerate(units):
            flattened_units = []
            flattened_units.append(unit["OBJ"])
            flattened_units.append(unit["VERB"])
            flattened_units.extend(unit["MOD"])
            
            # Calculate total possible combinations for this unit
            total_combinations = 1
            for unit_list in flattened_units:
                total_combinations *= len(unit_list)
            
            # For performance, limit combinations per unit to prevent memory explosion
            max_combinations_per_unit = min(total_combinations, 100)  # Hard cap for performance
            
            # Sample combinations if needed
            if total_combinations <= max_combinations_per_unit:
                unit_products_list = list(product(*flattened_units))
            else:
                unit_products_list = self._sample_combinations(flattened_units, max_combinations_per_unit)
            
            for product_idx, unit_product in enumerate(unit_products_list):
                mods = unit_product[2:]
                
                # Split modifiers into prefix/infix/suffix positions
                mod_ids = list(range(len(mods)))
                names = unit["SEQUENCE"]
                split_ids = divide_list_into_three(mod_ids)
                
                for split_idx, split_id in enumerate(split_ids):
                    prefix_ids, infix_ids, suffix_ids = split_id
                    prefix = [mods[i] for i in prefix_ids]
                    infix = [mods[i] for i in infix_ids]
                    
                    # Validate infix - only single, valid phrases allowed
                    if len(infix) > 1:
                        continue
                    if infix and len(infix) == 1:
                        # Resolve lazy object first for validation
                        resolved_infix_item = self._resolve_if_lazy(infix[0])
                        if resolved_infix_item is None or not isinstance(resolved_infix_item, PhraseEn):
                            continue
                    
                    suffix = [mods[i] for i in suffix_ids]
                    names_prefix = [names[i] for i in prefix_ids]
                    names_infix = [names[i] for i in infix_ids]
                    names_suffix = [names[i] for i in suffix_ids]
                    
                    # Format prefix with commas - resolve lazy objects first
                    if prefix:
                        resolved_prefix = [self._resolve_if_lazy(item) for item in prefix]
                        prefix = SP(*resolved_prefix).a(',')
                    else:
                        prefix = None
                    
                    # Resolve lazy objects in infix and suffix lists
                    if infix:
                        infix = [self._resolve_if_lazy(item) for item in infix]
                    if suffix:
                        suffix = [self._resolve_if_lazy(item) for item in suffix]
                    
                    # Format object content based on mode
                    formatted_obj = self._format_object_content(unit_product[0], obj_mode)
                    
                    # Generate sentence using template
                    curr_sent_cont, composed_sequence = template_f(
                        formatted_obj, 
                        unit_product[1], 
                        [prefix, names_prefix], 
                        [infix, names_infix], 
                        [suffix, names_suffix]
                    )
                    
                    # Direction validation for non-translate motions
                    motion_type = getattr(unit.get('motion', None), 'type', None)  # If available
                    if motion_type and motion_type != "translate":
                        look_up = None
                        if "direction" in composed_sequence:
                            look_up = 'direction'
                        elif "direction_magnitude" in composed_sequence:
                            look_up = 'direction_magnitude'
                        
                        if look_up:
                            direction_id = composed_sequence.index(look_up)
                            if direction_id == 0:
                                continue
                            prev_unit = composed_sequence[direction_id - 1]
                            if prev_unit not in ["object", "verb"]:
                                continue
                    
                    ## Group by template and unit ordering (sequence)
                    sequence_key = tuple(composed_sequence)  # Convert to tuple for dict key
                    group_key = (template_name, unit_idx, sequence_key)
                    
                    if group_key not in grouped_results:
                        grouped_results[group_key] = {
                            'sentences': [],
                            'labels': []
                        }
                    
                    grouped_results[group_key]['sentences'].append(curr_sent_cont)
                    grouped_results[group_key]['labels'].append(composed_sequence)
        
        ## Apply group-based sampling: sample within each (template, unit) group
        all_results = []
        all_labels = []
        
        if self.sampling_config.get("enabled", True):
            max_per_group = self.sampling_config.get("max_per_group", 2)
            
            for group_key, group_data in grouped_results.items():
                template_name, unit_idx, sequence_key = group_key
                sentences = group_data['sentences']
                labels = group_data['labels']
                
                if not sentences:
                    continue
                
                # Sample within this (template, unit) group
                if len(sentences) <= max_per_group:
                    # Take all sentences if within limit
                    all_results.extend(sentences)
                    all_labels.extend(labels)
                else:
                    # Sample from this group
                    sample_indices = random.sample(range(len(sentences)), max_per_group)
                    sampled_sentences = [sentences[i] for i in sample_indices]
                    sampled_labels = [labels[i] for i in sample_indices]
                    
                    all_results.extend(sampled_sentences)
                    all_labels.extend(sampled_labels)
        else:
            # No sampling - take all results
            for group_data in grouped_results.values():
                all_results.extend(group_data['sentences'])
                all_labels.extend(group_data['labels'])
        
        return all_results, all_labels
    
    
    def compose(self, motion: Motion, template_name: str, **kwargs) -> Tuple[List[S], List[List[str]]]:
        """
        Consolidated method to generate sentences for a motion - replaces the multi-step process.
        
        This method consolidates the functionality from:
        - generate_sentence_constituent()
        - _generate_all_unit_combinations() 
        - apply_verb_conflation()
        - generate_positioned_sentences()
        """
        
        # Generate base sentence components
        unit_generator = SentenceUnit(self.vocab)
        obj_phrases = unit_generator.generate_object_phrases(motion.agent)
        verb_alternatives = unit_generator.generate_verb_alternatives(motion)
        dir_components = unit_generator.generate_direction_components(motion)
        mag_components = unit_generator.generate_magnitude_components(motion)
        dur_phrases = unit_generator.generate_duration_phrases(motion)
        orig_phrases = unit_generator.generate_origin_phrases(motion)
        post_phrases = unit_generator.generate_post_phrases(motion)
        
        base_units = {
            "obj": obj_phrases,
            "verb": verb_alternatives,
            "dir": dir_components,
            "mag": mag_components,
            "dur": dur_phrases,
            "orig": orig_phrases,
            "post": post_phrases
        }
        
        # Generate all unit combinations with permutations
        all_units = self._generate_unit_combinations(base_units)
        
        # Apply verb conflation and add conflated units
        conflated_units = self.apply_verb_conflation(motion, base_units, dir_components, mag_components)
        all_units.extend(conflated_units)
        
        # Assemble units into sentences
        return self.assemble_units(all_units, template_name, **kwargs)
    

    def _sample_combinations(self, flattened_units: List[List], max_combinations: int) -> List[Tuple]:
        """
        Sample combinations without generating all combinations first.
        Uses reservoir sampling approach for memory efficiency.
        """
        import random
        
        # Calculate total possible combinations
        total_combinations = 1
        for unit_list in flattened_units:
            total_combinations *= len(unit_list)
        
        if total_combinations <= max_combinations:
            return list(product(*flattened_units))
        
        # Sample indices for each dimension
        sampled_combinations = []
        attempts = 0
        max_attempts = max_combinations * 3  # Avoid infinite loops
        
        while len(sampled_combinations) < max_combinations and attempts < max_attempts:
            # Generate random combination by sampling one element from each unit list
            combination = tuple(random.choice(unit_list) for unit_list in flattened_units)
            
            # Avoid duplicates (simple check)
            if combination not in sampled_combinations:
                sampled_combinations.append(combination)
            
            attempts += 1
        
        return sampled_combinations
    
    
    def compose_direction_magnitude(self, dir_conts: Dict, mag_conts: Dict) -> List[Any]:
        """Compose direction and magnitude - extracted from compose_dir_mag_conts()."""
        x_conts = []
        y_conts = []
        
        ## x components - resolve lazy objects before AdvP construction
        for x_dir_cont in dir_conts["x"]:
            for x_mag_cont in mag_conts["x"]:
                # Resolve lazy objects first
                resolved_x_dir = self._resolve_if_lazy(x_dir_cont)
                resolved_x_mag = self._resolve_if_lazy(x_mag_cont)
                
                if resolved_x_dir is None and resolved_x_mag is not None:
                    x_conts.append(AdvP(resolved_x_mag))
                elif resolved_x_dir is not None and resolved_x_mag is None:
                    x_conts.append(AdvP(resolved_x_dir))
                elif resolved_x_dir is not None and resolved_x_mag is not None:
                    x_conts.append(AdvP(resolved_x_dir, resolved_x_mag))
        
        ## y components - resolve lazy objects before AdvP construction
        for y_dir_cont in dir_conts["y"]:
            for y_mag_cont in mag_conts["y"]:
                # Resolve lazy objects first
                resolved_y_dir = self._resolve_if_lazy(y_dir_cont)
                resolved_y_mag = self._resolve_if_lazy(y_mag_cont)
                
                if resolved_y_dir is None and resolved_y_mag is not None:
                    y_conts.append(AdvP(resolved_y_mag))
                elif resolved_y_dir is not None and resolved_y_mag is None:
                    y_conts.append(AdvP(resolved_y_dir))
                elif resolved_y_dir is not None and resolved_y_mag is not None:
                    y_conts.append(AdvP(resolved_y_dir, resolved_y_mag))
        
        x_conts = [None] if len(x_conts) == 0 else x_conts
        y_conts = [None] if len(y_conts) == 0 else y_conts
        
        # Combine x and y components
        from pyrealb import CP, C
        dir_mag_conts = []
        for x_cont in x_conts:
            for y_cont in y_conts:
                if x_cont is None and y_cont is None:
                    dir_mag_conts.append(None)
                elif x_cont is None:
                    dir_mag_conts.append(y_cont)
                elif y_cont is None:
                    dir_mag_conts.append(x_cont)
                else:
                    dir_mag_conts.append(CP(C("and"), x_cont, y_cont))
        
        return dir_mag_conts
    
    
    def resolve_modifier_content(self, dict_cont: Any) -> List[Any]:
        """Resolve dictionary content"""
        def resolve_lazy_objects(content):
            """Recursively resolve LazyPyRealB objects in content."""
            if isinstance(content, LazyPyRealB):
                return content.get_pyrealb()
            elif isinstance(content, list):
                resolved_list = []
                for item in content:
                    if isinstance(item, LazyPyRealB):
                        resolved_list.append(item.get_pyrealb())
                    elif isinstance(item, list):
                        ## Handle nested lists (like the scaling direction case)
                        ## For lists with 2 elements [adv, axis], create a flattened representation
                        if len(item) == 2 and item[1] is not None and item[1] != "":
                            ## This is a [direction_adv, axis_modifier] pair - combine them
                            dir_part = resolve_lazy_objects(item[0]) if item[0] is not None else None
                            axis_part = resolve_lazy_objects(item[1]) if item[1] is not None else None
                            if dir_part is not None and axis_part is not None:
                                resolved_list.append([dir_part, axis_part])
                            elif dir_part is not None:
                                resolved_list.append(dir_part)
                            elif axis_part is not None:
                                resolved_list.append(axis_part)
                        else:
                            ## Regular nested list handling
                            resolved_sublist = resolve_lazy_objects(item)
                            if isinstance(resolved_sublist, list):
                                resolved_list.extend(resolved_sublist)
                            else:
                                resolved_list.append(resolved_sublist)
                    else:
                        resolved_list.append(item)
                return resolved_list
            else:
                return content
        
        if isinstance(dict_cont, dict):
            if dict_cont["x"] != [None] and dict_cont["y"] == [None]:
                result = dict_cont["x"]
            elif dict_cont["x"] == [None] and dict_cont["y"] != [None]:
                result = dict_cont["y"]
            else:
                result = [None]
                
            if isinstance(result, list) and len(result) == 0:
                return [None]
            result = resolve_lazy_objects(result)
            return result
        elif isinstance(dict_cont, list):
            if len(dict_cont) == 0:
                return [None]
            return resolve_lazy_objects(dict_cont)
        
        if isinstance(dict_cont, LazyPyRealB):
            return [dict_cont.get_pyrealb()]
        
        return dict_cont if dict_cont is not None else [None]
    
    
    def apply_verb_conflation(self, motion: Motion, base_units: Dict, dir_conts: Dict, mag_conts: Dict) -> List[Dict]:
        """Apply verb conflation - extracted from cross_unit_conts() conflation logic."""
        conflated_units = []
        motion_type = motion.type
        
        match motion_type:
            case "scale":
                if motion.direction:
                    x, y = motion.direction[0], motion.direction[1]
                    new_verb_conts = None
                    
                    # Both x and y scaled up
                    if np.sign(x) > 0 and np.sign(y) > 0:
                        try:
                            scale_up_verbs = self.vocab.get_vocab("motions", "scale", "VERB_UP")
                            new_verb_conts = scale_up_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    # Both x and y scaled down
                    elif np.sign(x) < 0 and np.sign(y) < 0:
                        try:
                            scale_down_verbs = self.vocab.get_vocab("motions", "scale", "VERB_DOWN")
                            new_verb_conts = scale_down_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    # Only one dimension scaled up
                    elif (np.sign(x) > 0 and np.sign(y) == 0) or (np.sign(x) == 0 and np.sign(y) > 0):
                        try:
                            scale_up_verbs = self.vocab.get_vocab("motions", "scale", "VERB_UP")
                            scale_up_single_verbs = self.vocab.get_vocab("motions", "scale", "VERB_UP_SINGLE")
                            new_verb_conts = scale_up_verbs.to_pyrealb_list() + scale_up_single_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    # Only one dimension scaled down
                    elif (np.sign(x) < 0 and np.sign(y) == 0) or (np.sign(x) == 0 and np.sign(y) < 0):
                        try:
                            scale_down_verbs = self.vocab.get_vocab("motions", "scale", "VERB_DOWN")
                            scale_down_single_verbs = self.vocab.get_vocab("motions", "scale", "VERB_DOWN_SINGLE")
                            new_verb_conts = scale_down_verbs.to_pyrealb_list() + scale_down_single_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    if new_verb_conts:
                        # Remove directional adverbs, preserve axis expressions (avoid deepcopy for performance)
                        new_dir_conts = {
                            "x": [phrase[1] if isinstance(phrase, list) else phrase for phrase in dir_conts["x"]],
                            "y": [phrase[1] if isinstance(phrase, list) else phrase for phrase in dir_conts["y"]], 
                            "x==y": dir_conts["x==y"]
                        }
                        
                        new_dir_mag = self.compose_direction_magnitude(new_dir_conts, mag_conts)
                        
                        # Create conflated units
                        post_phrases = base_units.get("post", [None])
                        orig_phrases = base_units.get("orig", [None])
                        dur_phrases = base_units.get("dur", [None])
                        
                        conflated_units.append({
                            "OBJ": base_units["obj"],
                            "VERB": new_verb_conts,
                            "MOD": [new_dir_mag, post_phrases, orig_phrases, dur_phrases],
                            "SEQUENCE": ["direction_magnitude", "post", "origin", "duration"]
                        })
                        conflated_units.append({
                            "OBJ": base_units["obj"],
                            "VERB": new_verb_conts,
                            "MOD": [new_dir_mag, post_phrases, dur_phrases, orig_phrases],
                            "SEQUENCE": ["direction_magnitude", "post", "duration", "origin"]
                        })
            
            case "rotate":
                origin = motion.origin
                if origin and origin == ["50%", "50%"]:
                    try:
                        rotate_center_verbs = self.vocab.get_vocab("motions", "rotate", "VERB_CENTER")
                        new_verb_conts = rotate_center_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        
                        dir_resolved = self.resolve_modifier_content(dir_conts)
                        mag_resolved = self.resolve_modifier_content(mag_conts)
                        post_phrases = base_units.get("post", [None])
                        dur_phrases = base_units.get("dur", [None])
                        
                        conflated_units.append({
                            "OBJ": base_units["obj"],
                            "VERB": new_verb_conts,
                            "MOD": [dir_resolved, mag_resolved, post_phrases, dur_phrases],
                            "SEQUENCE": ["direction", "magnitude", "post", "duration"]
                        })
                    except:
                        pass
            
            case "translate":
                if motion.direction:
                    x, y = motion.direction[0], motion.direction[1]
                    new_verb_conts = None
                    
                    # Direction upward
                    if y > 0 and x < y:
                        try:
                            translate_up_verbs = self.vocab.get_vocab("motions", "translate", "VERB_UP")
                            new_verb_conts = translate_up_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    # Direction downward
                    elif y < 0 and x > y:
                        try:
                            translate_down_verbs = self.vocab.get_vocab("motions", "translate", "VERB_DOWN")
                            new_verb_conts = translate_down_verbs.to_pyrealb_list()  # Convert lazy objects to PyRealB
                        except:
                            pass
                    
                    if new_verb_conts:
                        # Remove directional adverbs
                        new_dir_conts = {"x": [None], "y": [None], "x==y": False}
                        new_dir_mag = self.compose_direction_magnitude(new_dir_conts, mag_conts)
                        
                        orig_phrases = base_units.get("orig", [None])
                        post_phrases = base_units.get("post", [None])
                        dur_phrases = base_units.get("dur", [None])
                        
                        conflated_units.append({
                            "OBJ": base_units["obj"],
                            "VERB": new_verb_conts,
                            "MOD": [new_dir_mag, orig_phrases, post_phrases, dur_phrases],
                            "SEQUENCE": ["magnitude", "origin", "post", "duration"]
                        })
        
        return conflated_units
