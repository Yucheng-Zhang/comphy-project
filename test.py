from rwpde import rwla2d_sp

u = rwla2d_sp(10, 100, 0.1, 13203179)

u.rw_at(5, 5)

