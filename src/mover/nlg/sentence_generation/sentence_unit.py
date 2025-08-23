"""
Core sentence unit generators extracted from original implementation.
"""
import math
import numpy as np
from typing import List, Dict, Any, Optional
from pyrealb import NP, PP, V, N, A, D, Adv, P, CP, C, NO, VP, loadEn, addToLexicon
from mover.nlg.sentence_generation.vocab import Vocab, LazyPyRealB
from mover.nlg.data_classes import Motion, Object
from mover.nlg.utils import num_cont_no_zero, compute_x_y_mag


class SentenceUnit:
    """Generates individual sentence components from motion data."""
    
    def __init__(self, vocab: Vocab):
        self.vocab = vocab
        loadEn()
        # Add custom lexicon entries
        addToLexicon("px", {"N": {"tab": "n4", "cnt": False}})
        addToLexicon("over a period of", {"P": {}})
        addToLexicon("second", {"N": {"tab": "n1", "cnt": True}})
        addToLexicon("degree", {"N": {"tab": "n1", "cnt": True}})
    
    
    def generate_object_phrases(self, object_list: List[Object]) -> List[NP]:
        """Generate object noun phrases - extracted from unit_object()."""
        ## Get determiners from vocabulary
        det_vocab = self.vocab.get_vocab("objects", "determiners", "")
        det = det_vocab.to_pyrealb_list()[0] if len(det_vocab) > 0 else D("the")
        
        ## Single object
        if len(object_list) == 1:
            obj = object_list[0]
            return [NP(det, N(obj.shape), [A(obj.fill)])]
        
        ## Multiple objects
        obj_phrases = [NP(det, N(obj.shape), [A(obj.fill)]) for obj in object_list]
        return [CP(C("and"), *obj_phrases)]
    
    
    def generate_verb_alternatives(self, motion: Motion) -> List[V]:
        """Generate verb alternatives - extracted from unit_verb()."""
        motion_verbs = self.vocab.get_vocab("motions", motion.type, "VERB")
        return motion_verbs.to_pyrealb_list()
    
    
    def generate_direction_components(self, motion: Motion) -> Dict[str, Any]:
        """Generate direction components - extracted from unit_direction()."""
        direction = motion.direction
        dir_conts = {"x": [None], "y": [None], "x==y": False}
        
        if not direction:
            return dir_conts
        
        match motion.type:
            case "translate":
                x, y = direction[0], direction[1]
                
                if x > 0:
                    right_adv = self.vocab.get_vocab("directions", "right", "ADV")
                    right_phrase = self.vocab.get_vocab("directions", "right", "PHRASE")
                    dir_conts["x"] = right_adv.to_pyrealb_list() + right_phrase.to_pyrealb_list()
                elif x < 0:
                    left_adv = self.vocab.get_vocab("directions", "left", "ADV")
                    left_phrase = self.vocab.get_vocab("directions", "left", "PHRASE")
                    dir_conts["x"] = left_adv.to_pyrealb_list() + left_phrase.to_pyrealb_list()
                
                if y > 0:
                    up_adv = self.vocab.get_vocab("directions", "up", "ADV")
                    up_phrase = self.vocab.get_vocab("directions", "up", "PHRASE")
                    dir_conts["y"] = up_adv.to_pyrealb_list() + up_phrase.to_pyrealb_list()
                elif y < 0:
                    down_adv = self.vocab.get_vocab("directions", "down", "ADV")
                    down_phrase = self.vocab.get_vocab("directions", "down", "PHRASE")
                    dir_conts["y"] = down_adv.to_pyrealb_list() + down_phrase.to_pyrealb_list()
                    
            case "rotate":
                dir_conts["x==y"] = True
                if direction > 0:
                    cw_adv = self.vocab.get_vocab("directions", "clockwise", "ADV")
                    cw_phrase = self.vocab.get_vocab("directions", "clockwise", "PHRASE")
                    dir_conts["x"] = cw_adv.to_pyrealb_list() + cw_phrase.to_pyrealb_list()
                elif direction < 0:
                    ccw_adv = self.vocab.get_vocab("directions", "counterclockwise", "ADV")
                    ccw_phrase = self.vocab.get_vocab("directions", "counterclockwise", "PHRASE")
                    dir_conts["x"] = ccw_adv.to_pyrealb_list() + ccw_phrase.to_pyrealb_list()
                else:
                    raise ValueError("Rotation direction cannot be 0")
                    
            case "scale":
                x, y = direction[0], direction[1]
                up_adv = self.vocab.get_vocab("directions", "scale_up", "ADV")
                up_indicators = up_adv.to_pyrealb_list()
                
                down_adv = self.vocab.get_vocab("directions", "scale_down", "ADV")
                down_indicators = down_adv.to_pyrealb_list()
                
                if x == y:
                    dir_conts["x==y"] = True
                    if np.sign(x) > 0:
                        dir_indicators = up_indicators
                    elif np.sign(x) < 0:
                        dir_indicators = down_indicators
                    else:
                        dir_indicators = [None]
                        
                    dir_axis_adv = self.vocab.get_vocab("directions", "both-x-y", "ADV")
                    dir_axis_phrase = self.vocab.get_vocab("directions", "both-x-y", "PHRASE")
                    dir_axis = dir_axis_adv.to_pyrealb_list() + dir_axis_phrase.to_pyrealb_list() + [""]
                    dir_conts["x"] = [[adv, axis_cont] for adv in dir_indicators for axis_cont in dir_axis]
                else:
                    # Handle separate x and y directions for non-uniform scaling
                    x_dir_indicators = up_indicators if np.sign(x) > 0 else down_indicators if np.sign(x) < 0 else [None]
                    y_dir_indicators = up_indicators if np.sign(y) > 0 else down_indicators if np.sign(y) < 0 else [None]
                    
                    x_axis_adv = self.vocab.get_vocab("directions", "x-axis", "ADV")
                    x_axis_phrase = self.vocab.get_vocab("directions", "x-axis", "PHRASE")
                    y_axis_adv = self.vocab.get_vocab("directions", "y-axis", "ADV")
                    y_axis_phrase = self.vocab.get_vocab("directions", "y-axis", "PHRASE")
                    
                    x_axis_conts = x_axis_adv.to_pyrealb_list() + x_axis_phrase.to_pyrealb_list()
                    y_axis_conts = y_axis_adv.to_pyrealb_list() + y_axis_phrase.to_pyrealb_list()
                    
                    dir_conts["x"] = [[adv, axis_cont] for adv in x_dir_indicators if adv for axis_cont in x_axis_conts if axis_cont]
                    dir_conts["y"] = [[adv, axis_cont] for adv in y_dir_indicators if adv for axis_cont in y_axis_conts if axis_cont]
                    
                    if not dir_conts["x"]:
                        dir_conts["x"] = [None]
                    if not dir_conts["y"]:
                        dir_conts["y"] = [None]
        
        return dir_conts
    
    
    def generate_magnitude_components(self, motion: Motion) -> Dict[str, Any]:
        """Generate magnitude components - extracted from unit_magnitude()."""
        magnitude = motion.magnitude
        mag_conts = {"x": [None], "y": [None], "x==y": False}
        
        if magnitude is None:
            return mag_conts

        ## Get prepositions from vocabulary
        prep_vocab = self.vocab.get_vocab("magnitude", "all", "PREP")
        preps = prep_vocab.to_pyrealb_list()
        
        match motion.type:
            case "translate":            
                if motion.direction:
                    x, y = compute_x_y_mag(magnitude, motion.direction) if not isinstance(magnitude, list) else (magnitude[0], magnitude[1])

                    ## per-axis case (not considering diagonal case for now) - use lazy construction
                    pp_already_mag_f = lambda prep, num, unit=None: LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(num), unit))
                    mag_conts["x"] = [pp_already_mag_f(prep, x, N("px")) for prep in preps] if x != 0 else [None]
                    mag_conts["y"] = [pp_already_mag_f(prep, y, N("px")) for prep in preps] if y != 0 else [None]
                            
                ## if no direction
                else:
                    if not isinstance(magnitude, list):
                        mag_conts["x==y"] = True
                        mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(magnitude), N("px"))) for prep in preps]
                    else:
                        x, y = magnitude[0], magnitude[1]
                        mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(x), N("px"))) for prep in preps] if x != 0 else [None]
                        mag_conts["y"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(y), N("px"))) for prep in preps] if y != 0 else [None]
                
            case "rotate":
                mag_conts["x==y"] = True
                mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(magnitude), N("degree"))) for prep in preps]
                mag_conts["x"].append([LazyPyRealB.phrase("PP", P("through"), LazyPyRealB.phrase("NP", D("a"), N("angle"), P("of"), LazyPyRealB.phrase("NP", num_cont_no_zero(magnitude), N("degree"))))])
                
            case "scale":
                if isinstance(magnitude, list) and len(magnitude) == 2:
                    x, y = magnitude[0], magnitude[1]
                    if x == y:
                        mag_conts["x==y"] = True
                        mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(x))) for prep in preps]
                    else:
                        if x != 0:
                            mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(x))) for prep in preps]
                        if y != 0:
                            mag_conts["y"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(y))) for prep in preps]
                else:
                    # Single magnitude value for uniform scaling
                    mag_conts["x==y"] = True
                    mag_conts["x"] = [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", num_cont_no_zero(magnitude))) for prep in preps]
        
        return mag_conts
    
    
    def generate_duration_phrases(self, motion: Motion) -> List[Any]:
        """Generate duration phrases - extracted from unit_duration()."""
        duration = motion.duration
        
        if duration is None:
            return [None]
        
        ## Get duration prepositions from vocabulary
        duration_preps = self.vocab.get_vocab("temporal", "duration", "PREP")
        preps = duration_preps.to_pyrealb_list()
        dur_num_cont = num_cont_no_zero(duration)
        return [LazyPyRealB.phrase("PP", prep, LazyPyRealB.phrase("NP", dur_num_cont, N("second"))) for prep in preps]
    
    
    def generate_origin_phrases(self, motion: Motion) -> List[Any]:
        """Generate origin phrases - extracted from unit_origin()."""
        origin = motion.origin
        
        if origin is None:
            return [None]

        ## Motion-type specific prepositions from vocabulary
        if motion.type == "scale":
            prep_vocab = self.vocab.get_vocab("objects", "origin_prepositions", "scale")
        else:
            prep_vocab = self.vocab.get_vocab("objects", "origin_prepositions", "default")
        preps_pyrealb = prep_vocab.to_pyrealb_list()
        preps = [prep.realize() if hasattr(prep, 'realize') else str(prep) for prep in preps_pyrealb]
        
        ## Handle string origins
        if isinstance(origin, str):
            return [PP(P(prep), origin) for prep in preps]
        
        ## Handle list origins (coordinates or percentages)
        elif isinstance(origin, (list, tuple)) and len(origin) == 2:
            x, y = origin[0], origin[1]
            
            ## Handle percentage strings (like ["50%", "50%"])
            if isinstance(x, str) and isinstance(y, str):
                x_val = float(x.replace("%", ""))
                y_val = float(y.replace("%", ""))
                singular = "s" if len(motion.agent) == 1 else "p"
                pos_det_cont = D("my").pe(3).ow(singular)  ## "its" for singular, "their" for plural
                
                ## Get location nouns from vocabulary
                location_nouns = self.vocab.get_vocab("objects", "location_nouns", "")
                location_list = location_nouns.to_pyrealb_list()
                center_cont = location_list[0].n(singular) if len(location_list) > 0 else N("center").n(singular)
                corner_cont = location_list[1].n(singular) if len(location_list) > 1 else N("corner").n(singular)
                
                ## Get coordinate adjectives from vocabulary
                coord_adjs = self.vocab.get_vocab("objects", "coordinate_adjectives", "")
                coord_adj_list = coord_adjs.to_pyrealb_list()
                
                ## Map specific percentage combinations to semantic locations
                if x_val == 50 and y_val == 50:
                    return [PP(P(prep), NP(pos_det_cont, center_cont)) for prep in preps]
                elif x_val == 0 and y_val == 0:
                    return [PP(P(prep), NP(pos_det_cont, coord_adj_list[0], coord_adj_list[2], corner_cont)) for prep in preps]  # top left
                elif x_val == 100 and y_val == 0:
                    return [PP(P(prep), NP(pos_det_cont, coord_adj_list[0], coord_adj_list[3], corner_cont)) for prep in preps]  # top right
                elif x_val == 0 and y_val == 100:
                    return [PP(P(prep), NP(pos_det_cont, coord_adj_list[1], coord_adj_list[2], corner_cont)) for prep in preps]  # bottom left
                elif x_val == 100 and y_val == 100:
                    return [PP(P(prep), NP(pos_det_cont, coord_adj_list[1], coord_adj_list[3], corner_cont)) for prep in preps]  # bottom right
            
            ## Handle numeric coordinates
            elif isinstance(x, (float, int)) and isinstance(y, (float, int)):
                ## Get determiners and point noun from vocabulary
                det_vocab = self.vocab.get_vocab("objects", "determiners", "")
                location_nouns = self.vocab.get_vocab("objects", "location_nouns",
                                                       "")
                location_list = location_nouns.to_pyrealb_list()
                point_noun = location_list[2] if len(location_list) > 2 else N("point")
                det = det_vocab.to_pyrealb_list()[0] if len(det_vocab) > 0 else D("the")
                return [PP(P(prep), det, point_noun, NP(num_cont_no_zero(x).a(","), num_cont_no_zero(y)).en("(")) for prep in preps]
        
        return [None]
    
    
    def generate_post_phrases(self, motion: Motion) -> List[Any]:
        """Generate post-condition phrases - extracted from unit_post()."""
        post = motion.post
        post_reference = motion.post_reference
        
        if post is None:
            return [None]
        
        ## Generate reference object phrases if they exist (matching original logic)
        reference_object_list = None
        if post_reference:
            reference_object_list = self.generate_object_phrases(post_reference)
            
        ## Look up spatial vocabulary for the post concept
        post_phrases = self.vocab.get_vocab("spatial", post, "PHRASE")
        if len(post_phrases) > 0 and reference_object_list:
            ## Integrate reference object into spatial phrases
            obj = reference_object_list[0]
            integrated_phrases = []
            for phrase in post_phrases:
                ## Handle the special case of spatial phrases that need object integration
                if phrase.get("type") == "PP" and "np" in phrase and phrase["np"].get("prep2"):
                    ## This is a spatial phrase like: "to the right of"
                    ## We need to create: "to the right of [object]"
                    main_prep = P(phrase["prep"])
                    np_def = phrase["np"]
                    
                    ## Build the NP parts (det, adj/noun) without prep2
                    np_parts = []
                    if "det" in np_def:
                        np_parts.append(D(np_def["det"]))
                    if "adj" in np_def:
                        np_parts.append(A(np_def["adj"]))
                    if "noun" in np_def:
                        np_parts.append(N(np_def["noun"]))
                    
                    ## Create the spatial phrase: "to the right of [object]"
                    spatial_np = NP(*np_parts)
                    prep2 = P(np_def["prep2"])
                    integrated_phrases.append(PP(main_prep, spatial_np, prep2, obj))
                
                elif phrase.get("type") == "VP" and "pp" in phrase and phrase["pp"].get("prep2"):
                    ## This is a VP spatial phrase like: "be on the right of"
                    ## We need to create: "be on the right of [object]"
                    verb = V(phrase["verb"]["word"])
                    if "tense" in phrase["verb"]:
                        verb = verb.t(phrase["verb"]["tense"])
                    
                    pp_def = phrase["pp"]
                    pp_prep = P(pp_def["prep"])
                    
                    ## Build the NP parts without prep2
                    np_parts = []
                    if "det" in pp_def:
                        np_parts.append(D(pp_def["det"]))
                    if "adj" in pp_def:
                        np_parts.append(A(pp_def["adj"]))
                    if "noun" in pp_def:
                        np_parts.append(N(pp_def["noun"]))
                    
                    spatial_np = NP(*np_parts)
                    prep2 = P(pp_def["prep2"])
                    spatial_pp = PP(pp_prep, spatial_np, prep2, obj)
                    integrated_phrases.append(VP(verb, spatial_pp))
                
                else:
                    ## Regular case - convert the phrase and combine with object
                    phrase_obj = self.vocab._to_pyrealb(phrase)
                    if hasattr(phrase_obj, 'elements') and phrase_obj.elements:
                        ## If it's a VP or PP, just append the object
                        if hasattr(phrase_obj, 'elements') and hasattr(phrase_obj, 'realize') and 'VP' in str(type(phrase_obj)):
                            ## It's a VP-like object
                            integrated_phrases.append(VP(*phrase_obj.elements, obj))
                        elif hasattr(phrase_obj, 'elements') and len(phrase_obj.elements) >= 1:
                            ## It's a PP-like object
                            integrated_phrases.append(PP(*phrase_obj.elements, obj))
                        else:
                            integrated_phrases.append(PP(phrase_obj, obj))
                    else:
                        integrated_phrases.append(PP(phrase_obj, obj))
            return integrated_phrases
        
        ## Use default spatial prepositions
        default_prep_vocab = self.vocab.get_vocab("spatial", "default", "PREP")
        default_preps = default_prep_vocab.to_pyrealb_list()
        
        if reference_object_list:
            return [PP(prep, reference_object_list[0]) for prep in default_preps]
        else:
            return [PP(prep, N(post)) for prep in default_preps]
