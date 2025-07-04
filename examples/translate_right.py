o_1 = iota(Object, lambda o: color(o, "black") and shape(o, "square"))

exists(Motion, lambda m_1: type(m_1, "translate") and direction(m_1, [1.0, 0.0]) and magnitude(m_1, 100.0) and duration(m_1, 2.0) and agent(m_1, o_1))