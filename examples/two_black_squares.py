o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))
o_2 = iota(Object, lambda o: color(o, "black") and shape(o, "square") and not o_1)

m_1 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1))
m_2 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, -1.0]) and agent(m, o_1))
m_3 = iota(Motion, lambda m: type(m, "translate") and direction(m, [1.0, 0.0]) and agent(m, o_1) and not m_1)
m_4 = iota(Motion, lambda m: type(m, "translate") and direction(m, [0.0, 1.0]) and agent(m, o_2))

t_before(m_1, m_2)
t_after(m_3, m_2)