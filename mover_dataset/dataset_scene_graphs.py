from mover.nlg.data_classes import Object, Motion


object_1 = Object(
    shape="circle",
    fill="blue",
)

object_2 = Object(
    shape="square",
    fill="black",
)

object_3 = Object(
    shape="square",
    fill="blue",
)

object_4 = Object(
    shape="circle",
    fill="black",
)

motion_1 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = 100.0,
    direction = [0.0, 1.0],
    duration = 0.5,
    origin = None
)

motion_2 = Motion(
    type = "rotate",
    agent = [object_2],
    magnitude = 90.0,
    direction = -1.0,
    duration = 1.0,
    origin = ["0%", "100%"]
)

motion_3 = Motion(
    type = "scale",
    agent = [object_1],
    magnitude = [1.5, 1.5],
    direction = [1.0, 1.0],
    duration = 2.5,
    origin = [100.0, 200.0]
)

motion_4 = Motion(
    type = "scale",
    agent = [object_1],
    magnitude = [1.0, 1.5],
    direction = [1.0, 1.5],
    duration = 2.5,
    origin = ["50%", "50%"]
)

motion_5 = Motion(
    type = "scale",
    agent = [object_4],
    duration = 2.5,
    origin = ["100%", "0%"]
)

motion_6 = Motion(
    type = "scale",
    agent = [object_2, object_3, object_4],
    duration = 2.5,
    origin = ["50%", "50%"],
    direction = [1.2, 1.5],
    magnitude = [1.2, 1.5]
)

motion_7 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = [100.0, 100],
    direction = [1.0, 1.0],
    duration = 0.5,
    origin = None
)

motion_8 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = 100,
    direction = [1.0, 2.5],
    duration = 0.5,
    origin = None
)

motion_9 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = 238.0,
    direction = [1.0, 0.0],
    duration = 1.5,
    origin = None
)

motion_10 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = [100, 0],
    direction = [1, -1.0],
    duration = 0.5,
    origin = None
)

motion_11 = Motion(
    type = "scale",
    agent = [object_2, object_3, object_4],
    duration = 2.5,
    origin = ["50%", "50%"],
    direction = [1.5, 0.8],
    magnitude = [1.5, 0.8]
)

motion_12 = Motion(
    type = "scale",
    agent = [object_2, object_3, object_4],
    duration = 2.5,
    origin = ["50%", "50%"],
    direction = [1.5, 1.0],
    magnitude = [1.5, 1.0]
)

motion_13 = Motion(
    type = "rotate",
    agent = [object_1],
    magnitude = 90.0,
    direction = 1.0,
    duration = 2.0,
    origin = ["50%", "50%"]
)

motion_14 = Motion(
    type = "translate",
    agent = [object_2],
    magnitude = 100.0,
    direction = [0.0, -1.0],
    duration = 3.0,
    origin = None
)

motion_15 = Motion(
    type = "scale",
    agent = [object_3],
    magnitude = [0.0, 0.8],
    direction = [0.0, -1.0],
    duration = 2.5,
    origin = ["0%", "0%"]
)

motion_16 = Motion(
    type = "scale",
    agent = [object_1],
    magnitude = [1.5, 0.0],
    direction = [1.0, 0.0],
    duration = 4.0,
    origin = ["50%", "50%"]
)

motion_17 = Motion(
    type = "translate",
    agent = [object_3],
    magnitude = 36.0,
    direction = [-1.0, 0.0],
    duration = 15.0,
    origin = None
)

motion_18 = Motion(
    type = "scale",
    agent = [object_3],
    magnitude = [0.75, 0.75],
    direction = [-1.0, -1.0],
    duration = 0.75,
    origin = ["100%", "0%"]
)
    
motion_19 = Motion(
    type = "scale",
    agent = [object_3],
    magnitude = [0.0, 1.75],
    direction = [0.0, 1.0],
    duration = 0.75,
    origin = [0.0, 0.0]
)

motion_20 = Motion(
    type = "scale",
    agent = [object_3],
    magnitude = [0.8, 0.0],
    direction = [-1.0, 0.0],
    duration = 0.75,
    origin = [400.0, 400.0]
)

## spatial relations
motion_s_1 = Motion(
    type = "translate",
    agent = [object_1],
    post = "right",
    post_reference = [object_2],
    duration = 0.5,
)

motion_s_2 = Motion(
    type = "rotate",
    agent = [object_3],
    post = "intersecting",
    post_reference = [object_2],
    duration = 1.0,
)

motion_s_2_ref_4 = Motion(
    type = "rotate",
    agent = [object_3],
    post = "intersecting",
    post_reference = [object_4],
    duration = 1.0,
)

motion_s_3 = Motion(
    type = "scale",
    agent = [object_1],
    post = "bordering",
    post_reference = [object_3],
    direction = [1.0, 1.0],
)

motion_s_4 = Motion(
    type = "rotate",
    agent = [object_3],
    post = "bottom_left",
    post_reference = [object_1],
)

motion_s_5 = Motion(
    type = "translate",
    agent = [object_2],
    post = "top_bordering",
    post_reference = [object_4],
)

motion_s_6 = Motion(
    type = "translate",
    agent = [object_2],
    post = "bottom_bordering",
    post_reference = [object_3],
)

motion_s_7 = Motion(
    type = "rotate",
    agent = [object_3],
    post = "left_bordering",
    duration = 2.0,
    post_reference = [object_4],
)

motion_s_8 = Motion(
    type = "scale",
    agent = [object_2],
    post = "right_bordering",
    post_reference = [object_1],
)

motion_s_9 = Motion(
    type = "scale",
    agent = [object_2],
    post = "top_left",
    duration = 1.5,
    post_reference = [object_1],
)

motion_s_10 = Motion(
    type = "translate",
    agent = [object_2],
    post = "top_right",
    duration = 1.5,
    post_reference = [object_1],
)

motion_s_11 = Motion(
    type = "rotate",
    agent = [object_1],
    post = "bottom_right",
    post_reference = [object_2],
)

motion_s_12 = Motion(
    type = "rotate",
    agent = [object_1],
    post = "top",
    post_reference = [object_2],
)

motion_s_13 = Motion(
    type = "rotate",
    agent = [object_1],
    post = "bottom",
    duration = 1.5,
    post_reference = [object_2],
)

motion_s_13_t = Motion(
    type = "translate",
    agent = [object_1],
    post = "bottom",
    duration = 1.5,
    post_reference = [object_2],
)

motion_s_14 = Motion(
    type = "translate",
    agent = [object_3],
    post = "left",
    post_reference = [object_2],
)

motion_s_14_ref_4 = Motion(
    type = "translate",
    agent = [object_3],
    post = "left",
    post_reference = [object_4],
)


## for multi-motion testing
motion_m_1 = Motion(
    type = "translate",
    agent = [object_1],
    magnitude = 100.0,
    direction = [0.0, 1.0],
    duration = None,
    origin = None
)

motion_m_2 = Motion(
    type = "rotate",
    agent = [object_1],
    magnitude = 90.0,
    direction = -1.0,
    duration = 1.0,
    origin = None
)

motion_m_3 = Motion(
    type = "scale",
    agent = [object_2],
    magnitude = None,
    direction = [1.0, 1.0],
    duration = 2.5,
    origin = None
)

motion_m_4 = Motion(
    type = "translate",
    agent = [object_2],
    magnitude = 75.0,
    direction = [-1.0, 0.0],
    duration = 1.0,
    origin = None
)

motion_m_5 = Motion(
    type = "rotate",
    agent = [object_3],
    magnitude = 180.0,
    direction = -1.0,
    duration = None,
    origin = ["50%", "50%"]
)

## multi-motion: before(translate, rotate)
gen_multi_before_t_r = {
    "motions": [motion_m_1, motion_m_2],
    "relations": [None, "before"],
    "file_name": "test_1222_multi_before_t_r.json"
}

gen_multi_before_r_t = {
    "motions": [motion_m_2, motion_m_1],
    "relations": [None, "before"],
    "file_name": "test_1222_multi_before_r_t.json"
}

gen_multi_before_t_s = {
    "motions": [motion_m_4, motion_m_3],
    "relations": [None, "before"],
    "file_name": "test_1222_multi_before_t_s.json"
}

gen_multi_before_s_t = {
    "motions": [motion_m_3, motion_m_4],
    "relations": [None, "before"],
    "file_name": "test_1222_multi_before_s_t.json"
}

## multi-motion: while(scale, translate)
gen_multi_while_s_t = {
    "motions": [motion_m_3, motion_m_4],
    "relations": [None, "overlaps"],
    "file_name": "test_1222_multi_while_s_t.json"
}

gen_multi_while_t_s = {
    "motions": [motion_m_4, motion_m_3],
    "relations": [None, "overlaps"],
    "file_name": "test_1222_multi_while_t_s.json"
}

gen_multi_while_r_t = {
    "motions": [motion_m_5, motion_m_4],
    "relations": [None, "overlaps"],
    "file_name": "test_1222_multi_while_r_t.json"
}

gen_multi_while_t_r = {
    "motions": [motion_m_4, motion_m_5],
    "relations": [None, "overlaps"],
    "file_name": "test_1222_multi_while_t_r.json"
}


## multi-motion: after(scale, rotate)
gen_multi_after_s_r = {
    "motions": [motion_m_3, motion_m_5],
    "relations": [None, "after"],
    "file_name": "test_1222_multi_after_s_r.json"
}

gen_multi_after_r_s = {
    "motions": [motion_m_5, motion_m_3],
    "relations": [None, "after"],
    "file_name": "test_1222_multi_after_r_s.json"
}

gen_multi_after_t_s = {
    "motions": [motion_m_1, motion_m_3],
    "relations": [None, "after"],
    "file_name": "test_1222_multi_after_t_s.json"
}

gen_multi_after_s_t = {
    "motions": [motion_m_3, motion_m_1],
    "relations": [None, "after"],
    "file_name": "test_1222_multi_after_s_t.json"
}




###### multi-motion with spatial relations - before
gen_st_before_top = {
    "motions": [motion_s_12, motion_m_1],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_top.json"
}

gen_st_before_bottom = {
    "motions": [motion_s_13_t, motion_m_2],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_bottom.json"
}

gen_st_before_left = {
    "motions": [motion_s_14, motion_m_3],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_left.json"
}

gen_st_before_right = {
    "motions": [motion_s_1, motion_m_5],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_right.json"
}

gen_st_before_touch = {
    "motions": [motion_s_2, motion_m_4],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_touch.json"
}

gen_st_before_border = {
    "motions": [motion_s_3, motion_m_1],
    "relations": [None, "before"],
    "file_name": "test_0117_st_before_border.json"
}


## multi-motion with spatial relations - after
gen_st_after_top = {
    "motions": [motion_m_1, motion_s_12],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_top.json"
}

gen_st_after_bottom = {
    "motions": [motion_m_2, motion_s_13_t],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_bottom.json"
}

gen_st_after_left = {
    "motions": [motion_m_3, motion_s_14],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_left.json"
}

gen_st_after_right = {
    "motions": [motion_m_5, motion_s_1],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_right.json"
}

gen_st_after_touch = {
    "motions": [motion_m_4, motion_s_2],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_touch.json"
}

gen_st_after_border = {
    "motions": [motion_m_1, motion_s_3],
    "relations": [None, "after"],
    "file_name": "test_0117_st_after_border.json"
}


## multi-motion with spatial relations - while
gen_st_while_top = {
    "motions": [motion_5, motion_s_12],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_top.json"
}

gen_st_while_bottom = {
    "motions": [motion_5, motion_s_13_t],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_bottom.json"
}

gen_st_while_left = {
    "motions": [motion_5, motion_s_14],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_left.json"
}

gen_st_while_right = {
    "motions": [motion_5, motion_s_1],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_right.json"
}

gen_st_while_touch = {
    "motions": [motion_5, motion_s_2],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_touch.json"
}

gen_st_while_border = {
    "motions": [motion_5, motion_s_3],
    "relations": [None, "overlaps"],
    "file_name": "test_0117_st_while_border.json"
}

#########################################


## translate up
gen_translate_up = {
    "motions": [motion_1],
    "relations": [None],
    "file_name": "test_1219_translate_up.json"
}


## translate down
gen_translate_down = {
    "motions": [motion_14],
    "relations": [None],
    "file_name": "test_1223_translate_down.json"
}


## translate left
gen_translate_left = {
    "motions": [motion_17],
    "relations": [None],
    "file_name": "test_1224_translate_left.json"
}

## translate right
gen_translate_right = {
    "motions": [motion_9],
    "relations": [None],
    "file_name": "test_1224_translate_right.json"
}

## scale up
gen_scale_up_uniform = {
    "motions": [motion_3],
    "relations": [None],
    "file_name": "test_1222_scale_up_uniform.json"
}

## scale down
gen_scale_down_uniform = {
    "motions": [motion_18],
    "relations": [None],
    "file_name": "test_1224_scale_down_uniform.json"
}

## scale up x
gen_scale_up_x = {
    "motions": [motion_16],
    "relations": [None],
    "file_name": "test_1224_scale_up_x.json"
}

## scale up y
gen_scale_up_y = {
    "motions": [motion_19],
    "relations": [None],
    "file_name": "test_1224_scale_up_y.json"
}

## scale down x
gen_scale_down_x = {
    "motions": [motion_20],
    "relations": [None],
    "file_name": "test_1224_scale_down_x.json"
}

## scale down y
gen_scale_down_y = {
    "motions": [motion_15],
    "relations": [None],
    "file_name": "test_1223_scale_down_y.json"
}

## rotate from self center
gen_rotate_self_center = {
    "motions": [motion_13],
    "relations": [None],
    "file_name": "test_1222_rotate_self_center.json"
}

## rotate bottom left
gen_rotate_bottom_left = {
    "motions": [motion_2],
    "relations": [None],
    "file_name": "test_1222_rotate_bottom_left.json"
}



## spatial: translate to right
gen_s_translate_to_right = {
    "motions": [motion_s_1],
    "relations": [None],
    "file_name": "test_0109_s_translate_to_right.json"
}

## spatial: rotate to intersecting
gen_s_rotate_to_intersect = {
    "motions": [motion_s_2_ref_4],
    "relations": [None],
    "file_name": "test_0109_s_rotate_to_intersecting.json"
}

## spatial: scale to bordering
gen_s_scale_to_border = {
    "motions": [motion_s_3],
    "relations": [None],
    "file_name": "test_0109_s_scale_to_bordering.json"
}

## spatial: rotate to bottom left
gen_s_rotate_to_bottom_left = {
    "motions": [motion_s_4],
    "relations": [None],
    "file_name": "test_0109_s_rotate_to_bottom_left.json"
}

## spatial: translate to top bordering
gen_s_translate_to_top_border = {
    "motions": [motion_s_5],
    "relations": [None],
    "file_name": "test_0109_s_translate_to_top_bordering.json"
}

## spatial: translate to bottom bordering
gen_s_translate_bottom_border = {
    "motions": [motion_s_6],
    "relations": [None],
    "file_name": "test_0109_s_translate_bottom_border.json"
}

## spatial: rotate to left bordering
gen_s_rotate_to_left_bordering = {
    "motions": [motion_s_7],
    "relations": [None],
    "file_name": "test_0109_s_rotate_left_border.json"
}

## spatial: scale to right bordering
gen_s_scale_to_right_bordering = {
    "motions": [motion_s_8],
    "relations": [None],
    "file_name": "test_0109_s_scale_to_right_bordering.json"
}

## spatial: scale to top left
gen_s_scale_to_top_left = {
    "motions": [motion_s_9],
    "relations": [None],
    "file_name": "test_0109_s_scale_to_top_left.json"
}

## spatial: translate to top right
gen_s_translate_to_top_right = {
    "motions": [motion_s_10],
    "relations": [None],
    "file_name": "test_0109_s_translate_to_top_right.json"
}

## spatial: rotate to bottom right
gen_s_rotate_to_bottom_right = {
    "motions": [motion_s_11],
    "relations": [None],
    "file_name": "test_0109_s_rotate_to_bottom_right.json"
}

## spatial: rotate to top
gen_s_rotate_to_top = {
    "motions": [motion_s_12],
    "relations": [None],
    "file_name": "test_0109_s_rotate_to_top.json"
}

## spatial: rotate to bottom
gen_s_rotate_to_bottom = {
    "motions": [motion_s_13],
    "relations": [None],
    "file_name": "test_0109_s_rotate_to_bottom.json"
}

## spatial: translate to left
gen_s_translate_to_left = {
    "motions": [motion_s_14_ref_4],
    "relations": [None],
    "file_name": "test_0109_s_translate_to_left.json"
}


## 12 base prompts
gen_data_all_atomic = [gen_translate_up, gen_translate_down, gen_translate_left, gen_translate_right, gen_scale_up_uniform, gen_scale_down_uniform, gen_scale_up_x, gen_scale_up_y, gen_scale_down_x, gen_scale_down_y, gen_rotate_self_center, gen_rotate_bottom_left]

## 14 spatial prompts
gen_data_all_spatial = [gen_s_translate_to_right, gen_s_rotate_to_intersect, gen_s_scale_to_border, gen_s_rotate_to_bottom_left, gen_s_translate_to_top_border, gen_s_translate_bottom_border, gen_s_rotate_to_left_bordering, gen_s_scale_to_right_bordering, gen_s_scale_to_top_left, gen_s_translate_to_top_right, gen_s_rotate_to_bottom_right, gen_s_rotate_to_top, gen_s_rotate_to_bottom, gen_s_translate_to_left]

## 12 multi-motion prompt
gen_data_all_temporal = [gen_multi_before_t_r, gen_multi_before_r_t, gen_multi_before_t_s, gen_multi_before_s_t, gen_multi_while_s_t, gen_multi_while_t_s, gen_multi_while_r_t, gen_multi_while_t_r, gen_multi_after_s_r, gen_multi_after_r_s, gen_multi_after_t_s, gen_multi_after_s_t]

## 18 spatial temporal prompt
gen_data_all_spatial_temporal = [gen_st_before_top, gen_st_before_bottom, gen_st_before_left, gen_st_before_right, gen_st_before_touch, gen_st_before_border, gen_st_after_top, gen_st_after_bottom, gen_st_after_left, gen_st_after_right, gen_st_after_touch, gen_st_after_border, gen_st_while_top, gen_st_while_bottom, gen_st_while_left, gen_st_while_right, gen_st_while_touch, gen_st_while_border]


gen_data_all = gen_data_all_atomic + gen_data_all_spatial + gen_data_all_temporal + gen_data_all_spatial_temporal