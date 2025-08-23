from typing import Dict, Any
from mover.nlg.data_classes import Predicate


class MoVerFOLGenerator:
    """
    Generate MoVer program from motion scene graph.
    """
    
    def generate_program(self, gen_data: Dict[str, Any]) -> str:
        """Generate complete FOL program from motion data."""
        motions = gen_data["motions"]
        relations = gen_data["relations"]
        
        # Extract unique objects
        objects = []
        for motion in motions:
            if motion.agent:
                for obj in motion.agent:
                    if obj not in objects:
                        objects.append(obj)
            if motion.post_reference:
                for obj in motion.post_reference:
                    if obj not in objects:
                        objects.append(obj)
        
        # Generate program parts
        object_defs = []
        motion_defs = []
        temporal_rels = []
        
        # Object definitions
        for i, obj in enumerate(objects):
            var = f"o_{i+1}"
            predicates = []
            if obj.fill:
                predicates.append(Predicate("color", ["o"], [], [obj.fill]))
            if obj.shape:
                predicates.append(Predicate("shape", ["o"], [], [obj.shape]))
            
            expr = " and ".join([pred.render() for pred in predicates])
            object_defs.append(f"{var} = iota(Object, lambda o: {expr})")
        
        # Motion definitions
        for i, motion in enumerate(motions):
            var = f"m_{i+1}"
            
            # Determine if we use exists or iota (following gen_data.py logic)
            use_exists = (relations[i] is None and 
                         (i + 1 >= len(relations) or relations[i + 1] is None))
            
            # Set lambda variable: use actual var for exists, "m" for iota
            lambda_var = var if use_exists else "m"
            
            predicates = []
            
            if motion.type:
                predicates.append(Predicate("type", [lambda_var], [], [motion.type]))
            
            if motion.direction is not None:
                if motion.type == "rotate":
                    direction = "clockwise" if motion.direction > 0 else "counterclockwise"
                    predicates.append(Predicate("direction", [lambda_var], [], [direction]))
                else:
                    predicates.append(Predicate("direction", [lambda_var], [], [motion.direction]))
            
            if motion.magnitude is not None:
                predicates.append(Predicate("magnitude", [lambda_var], [], [motion.magnitude]))
            
            if motion.origin is not None:
                predicates.append(Predicate("origin", [lambda_var], [], [motion.origin]))
            
            if motion.post and motion.post_reference:
                ref_obj = motion.post_reference[0]
                ref_var = f"o_{objects.index(ref_obj) + 1}"
                agent_var = f"o_{objects.index(motion.agent[0]) + 1}"
                spatial_type = motion.post.replace("ing", "")
                spatial_pred = Predicate(f"s_{spatial_type}", [agent_var], [ref_var], []).render()
                predicates.append(Predicate("post", [lambda_var], [], [{"str": spatial_pred}]))
            
            if motion.duration is not None:
                predicates.append(Predicate("duration", [lambda_var], [], [motion.duration]))
            
            if motion.agent:
                agent_var = f"o_{objects.index(motion.agent[0]) + 1}"
                predicates.append(Predicate("agent", [lambda_var], [agent_var], []))
            
            expr = " and ".join([pred.render() for pred in predicates])
            
            if use_exists:
                motion_defs.append(f"exists(Motion, lambda {var}: {expr})")
            else:
                motion_defs.append(f"{var} = iota(Motion, lambda m: {expr})")
        
        # Temporal relations
        for i, relation in enumerate(relations):
            if i == 0 or relation is None:
                continue
            
            prev_var = f"m_{i}"
            curr_var = f"m_{i+1}"
            
            if relation == "before":
                temporal_rels.append(Predicate("t_before", [prev_var, curr_var], [], []).render())
            elif relation == "overlaps":
                temporal_rels.append(Predicate("t_while", [prev_var, curr_var], [], []).render())
            elif relation == "after":
                temporal_rels.append(Predicate("t_after", [prev_var, curr_var], [], []).render())
        
        # Combine all parts
        program_parts = []
        if object_defs:
            program_parts.append("\n".join(object_defs))
        if motion_defs:
            program_parts.append("\n".join(motion_defs))
        if temporal_rels:
            program_parts.append("\n".join(temporal_rels))
        
        return "\n\n".join(program_parts) 