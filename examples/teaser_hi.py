o_1 = iota(Object, lambda o: shape(o, "circle") and color(o, "orange"))
o_2 = iota(Object, lambda o: shape(o, "rectangle"))
o_3 = iota(Object, lambda o: id(o, "letter-H"))

m_1 = iota(Motion, lambda m: type(m, "translate") and post(m, s_top(o_1, o_2)) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "rotate") and direction(m, "clockwise") and magnitude(m, 90.0) and agent(m, o_3))

t_while(m_1, m_2)