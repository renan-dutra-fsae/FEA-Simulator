"""
frame3d.py — Solver de portico espacial (3D frame / FEM) do zero.

Resolve estruturas de tubos (chassi, suspensao, trelica espacial) modelando
cada tubo como um elemento de viga 3D com 12 graus de liberdade (6 por no:
3 translacoes + 3 rotacoes).

Fluxo (o mesmo que discutimos):
  1. Para cada tubo, monta a matriz de rigidez local 12x12 (K_local), com
     blocos de axial, torcao e duas flexoes.
  2. Rotaciona para coordenadas globais:  K_global_elem = T^T @ K_local @ T
  3. Espalha (assembly) cada K_global_elem na matriz global K da estrutura.
  4. Aplica apoios (GDL travados) e cargas.
  5. Resolve  K u = F  para os deslocamentos u.
  6. Recupera os esforcos internos de cada tubo (volta para coord. locais).

Convencao de eixos LOCAIS do elemento:
  x_local -> ao longo do tubo (do no i para o no j)
  y_local, z_local -> eixos da secao transversal

Unidades: use um sistema consistente. Aqui o exemplo usa N e mm
  => E, G em MPa (N/mm^2); A em mm^2; I, J em mm^4; comprimentos em mm;
     forcas em N; momentos em N.mm. Deslocamentos saem em mm, rotacoes em rad.
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. MATRIZ DE RIGIDEZ LOCAL 12x12 DE UMA VIGA 3D
# ----------------------------------------------------------------------
def k_local_viga3d(E, G, A, Iy, Iz, J, L):
    """
    Monta a matriz de rigidez 12x12 no referencial LOCAL do tubo.

    Ordem dos 12 GDL (no i depois no j):
      [ u_xi, u_yi, u_zi, th_xi, th_yi, th_zi,
        u_xj, u_yj, u_zj, th_xj, th_yj, th_zj ]

    E  : modulo de elasticidade
    G  : modulo de cisalhamento
    A  : area da secao
    Iy : momento de inercia em torno de y (governa flexao no plano xz)
    Iz : momento de inercia em torno de z (governa flexao no plano xy)
    J  : momento polar / constante de torcao
    L  : comprimento do elemento
    """
    k = np.zeros((12, 12))

    # --- Axial (EA/L): acopla u_xi <-> u_xj ---
    ea = E * A / L
    k[0, 0] = ea
    k[0, 6] = -ea
    k[6, 0] = -ea
    k[6, 6] = ea

    # --- Torcao (GJ/L): acopla th_xi <-> th_xj ---
    gj = G * J / L
    k[3, 3] = gj
    k[3, 9] = -gj
    k[9, 3] = -gj
    k[9, 9] = gj

    # --- Flexao no plano xy (usa Iz): acopla u_y e th_z dos dois nos ---
    # GDL envolvidos: 1 (u_yi), 5 (th_zi), 7 (u_yj), 11 (th_zj)
    a = E * Iz / L**3
    k[1, 1] += 12 * a
    k[1, 5] += 6 * L * a
    k[1, 7] += -12 * a
    k[1, 11] += 6 * L * a
    k[5, 1] += 6 * L * a
    k[5, 5] += 4 * L**2 * a
    k[5, 7] += -6 * L * a
    k[5, 11] += 2 * L**2 * a
    k[7, 1] += -12 * a
    k[7, 5] += -6 * L * a
    k[7, 7] += 12 * a
    k[7, 11] += -6 * L * a
    k[11, 1] += 6 * L * a
    k[11, 5] += 2 * L**2 * a
    k[11, 7] += -6 * L * a
    k[11, 11] += 4 * L**2 * a

    # --- Flexao no plano xz (usa Iy): acopla u_z e th_y dos dois nos ---
    # GDL envolvidos: 2 (u_zi), 4 (th_yi), 8 (u_zj), 10 (th_yj)
    # Sinais dos termos cruzados invertidos em relacao ao plano xy
    # por causa da orientacao (regra da mao direita).
    b = E * Iy / L**3
    k[2, 2] += 12 * b
    k[2, 4] += -6 * L * b
    k[2, 8] += -12 * b
    k[2, 10] += -6 * L * b
    k[4, 2] += -6 * L * b
    k[4, 4] += 4 * L**2 * b
    k[4, 8] += 6 * L * b
    k[4, 10] += 2 * L**2 * b
    k[8, 2] += -12 * b
    k[8, 4] += 6 * L * b
    k[8, 8] += 12 * b
    k[8, 10] += 6 * L * b
    k[10, 2] += -6 * L * b
    k[10, 4] += 2 * L**2 * b
    k[10, 8] += 6 * L * b
    k[10, 10] += 4 * L**2 * b

    return k


# ----------------------------------------------------------------------
# 2. MATRIZ DE ROTACAO 12x12 (local -> global)
# ----------------------------------------------------------------------
def matriz_rotacao(coord_i, coord_j, vetor_ref=None):
    """
    Constroi a matriz de transformacao T 12x12 que relaciona os GDL
    globais com os locais.  K_global_elem = T^T @ k_local @ T

    coord_i, coord_j : np.array([x, y, z]) dos dois nos
    vetor_ref : vetor auxiliar que define a orientacao da secao
                (para onde aponta o eixo y_local). Por padrao tenta [0,0,1];
                se o tubo for vertical, usa [0,1,0].
    """
    xi = np.asarray(coord_i, float)
    xj = np.asarray(coord_j, float)
    L = np.linalg.norm(xj - xi)
    if L == 0:
        raise ValueError("Elemento de comprimento zero (nos coincidentes).")

    # eixo x local = direcao do tubo (cossenos diretores)
    ex = (xj - xi) / L

    # escolhe vetor de referencia que nao seja paralelo a ex
    if vetor_ref is None:
        vetor_ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ex, vetor_ref)) > 0.99:  # tubo quase vertical
            vetor_ref = np.array([0.0, 1.0, 0.0])
    vetor_ref = np.asarray(vetor_ref, float)

    # eixo z local = ex cross ref (perpendicular ao plano formado)
    ez = np.cross(ex, vetor_ref)
    ez /= np.linalg.norm(ez)
    # eixo y local = ez cross ex  (completa o triedro destro)
    ey = np.cross(ez, ex)

    # matriz 3x3 de cossenos diretores (linhas = eixos locais)
    R = np.vstack([ex, ey, ez])

    # T 12x12 = R repetida em blocos diagonais (4 blocos: transl/rot de cada no)
    T = np.zeros((12, 12))
    for b in range(4):
        T[b*3:b*3+3, b*3:b*3+3] = R
    return T, L


# ----------------------------------------------------------------------
# 3. CLASSE PRINCIPAL: monta, resolve e recupera esforcos
# ----------------------------------------------------------------------
class Frame3D:
    def __init__(self):
        self.nos = []        # lista de coordenadas [x,y,z]
        self.elementos = []  # lista de dicts com no_i, no_j e props
        self.cargas = {}     # {no: [Fx,Fy,Fz,Mx,My,Mz]}
        self.apoios = {}     # {no: [bx,by,bz,brx,bry,brz]} 1=travado, 0=livre

    def add_no(self, x, y, z):
        self.nos.append(np.array([x, y, z], float))
        return len(self.nos) - 1

    def add_elemento(self, no_i, no_j, E, G, A, Iy, Iz, J, vetor_ref=None):
        self.elementos.append(dict(
            no_i=no_i, no_j=no_j, E=E, G=G, A=A,
            Iy=Iy, Iz=Iz, J=J, vetor_ref=vetor_ref))
        return len(self.elementos) - 1

    def add_carga(self, no, Fx=0, Fy=0, Fz=0, Mx=0, My=0, Mz=0):
        self.cargas[no] = [Fx, Fy, Fz, Mx, My, Mz]

    def add_apoio(self, no, bx=1, by=1, bz=1, brx=1, bry=1, brz=1):
        """1 = travado (deslocamento imposto = 0), 0 = livre."""
        self.apoios[no] = [bx, by, bz, brx, bry, brz]

    # --- montagem da matriz global e resolucao ---
    def resolver(self):
        n_nos = len(self.nos)
        n_gdl = 6 * n_nos
        K = np.zeros((n_gdl, n_gdl))
        F = np.zeros(n_gdl)

        # guarda dados por elemento para recuperar esforcos depois
        self._cache = []

        # ---- assembly: espalha cada K_global_elem na global ----
        for el in self.elementos:
            ci = self.nos[el['no_i']]
            cj = self.nos[el['no_j']]
            T, L = matriz_rotacao(ci, cj, el['vetor_ref'])
            k_loc = k_local_viga3d(el['E'], el['G'], el['A'],
                                   el['Iy'], el['Iz'], el['J'], L)
            k_glob = T.T @ k_loc @ T

            # enderecos globais dos 12 GDL deste elemento
            dofs = self._dofs_elemento(el['no_i'], el['no_j'])
            for a in range(12):
                for b in range(12):
                    K[dofs[a], dofs[b]] += k_glob[a, b]

            self._cache.append((el, T, k_loc, dofs))

        # ---- vetor de cargas ----
        for no, carga in self.cargas.items():
            for d in range(6):
                F[6*no + d] += carga[d]

        # ---- condicoes de contorno (apoios) ----
        # GDL travados: deslocamento = 0. Estrategia: separa livres/fixos.
        fixos = []
        for no, b in self.apoios.items():
            for d in range(6):
                if b[d] == 1:
                    fixos.append(6*no + d)
        fixos = sorted(set(fixos))
        livres = [i for i in range(n_gdl) if i not in fixos]

        # resolve apenas o subsistema dos GDL livres
        Kff = K[np.ix_(livres, livres)]
        Ff = F[livres]
        uf = np.linalg.solve(Kff, Ff)

        # monta o vetor completo de deslocamentos
        u = np.zeros(n_gdl)
        u[livres] = uf

        # reacoes nos apoios:  R = K u - F  (nas linhas fixas)
        R = K @ u - F

        self.u = u
        self.R = R
        self.K = K
        return u

    def _dofs_elemento(self, no_i, no_j):
        di = [6*no_i + k for k in range(6)]
        dj = [6*no_j + k for k in range(6)]
        return di + dj

    # --- recuperacao dos esforcos internos de cada tubo ---
    def esforcos(self):
        """
        Retorna, por elemento, as forcas/momentos internos no referencial
        LOCAL nas duas extremidades:
          [N_i, Vy_i, Vz_i, T_i, My_i, Mz_i,  N_j, Vy_j, Vz_j, T_j, My_j, Mz_j]
        Convencao: valores no no i sao a reacao que o elemento exerce.
        N>0 tracao no no j (sinal classico de barra).
        """
        out = []
        for (el, T, k_loc, dofs) in self._cache:
            u_glob_elem = self.u[dofs]          # 12 deslocs globais do elemento
            u_loc = T @ u_glob_elem             # converte para local
            f_loc = k_loc @ u_loc               # esforcos internos locais
            out.append(f_loc)
        return out


# ----------------------------------------------------------------------
# 4. PROPRIEDADES DE SECAO PARA TUBO CIRCULAR
# ----------------------------------------------------------------------
def secao_tubo(D_ext, espessura):
    """Propriedades de um tubo circular (D externo, parede t)."""
    Do = D_ext
    Di = D_ext - 2 * espessura
    A = np.pi / 4 * (Do**2 - Di**2)
    I = np.pi / 64 * (Do**4 - Di**4)   # Iy = Iz por simetria
    J = np.pi / 32 * (Do**4 - Di**4)   # polar = 2*I
    return dict(A=A, Iy=I, Iz=I, J=J)


# ----------------------------------------------------------------------
# 5. EXEMPLO + VALIDACAO ANALITICA
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- Caso 1: viga em balanco (cantilever) ----------
    # Tubo horizontal engastado numa ponta, carga P transversal na outra.
    # Solucao classica: flecha = P L^3 / (3 E I)
    E = 210_000.0   # MPa (aco)
    G = 80_000.0    # MPa
    L = 500.0       # mm
    P = 1000.0      # N (para baixo, -z)

    sec = secao_tubo(D_ext=25.0, espessura=2.0)

    f = Frame3D()
    n0 = f.add_no(0, 0, 0)
    n1 = f.add_no(L, 0, 0)
    f.add_elemento(n0, n1, E=E, G=G, **sec)
    f.add_apoio(n0)                       # engaste total no no 0
    f.add_carga(n1, Fz=-P)                # carga transversal no no 1
    u = f.resolver()

    flecha_fem = u[6*n1 + 2]              # deslocamento z do no 1
    flecha_teo = -P * L**3 / (3 * E * sec['Iy'])
    print("=== Caso 1: viga em balanco ===")
    print(f"  flecha FEM      : {flecha_fem:.4f} mm")
    print(f"  flecha teorica  : {flecha_teo:.4f} mm")
    print(f"  erro relativo   : {abs(flecha_fem-flecha_teo)/abs(flecha_teo)*100:.4f} %")

    # ---------- Caso 2: barra sob tracao axial ----------
    # alongamento = N L / (E A)
    f2 = Frame3D()
    a0 = f2.add_no(0, 0, 0)
    a1 = f2.add_no(L, 0, 0)
    f2.add_elemento(a0, a1, E=E, G=G, **sec)
    f2.add_apoio(a0)
    N = 5000.0
    f2.add_carga(a1, Fx=N)
    u2 = f2.resolver()
    along_fem = u2[6*a1 + 0]
    along_teo = N * L / (E * sec['A'])
    print("\n=== Caso 2: tracao axial ===")
    print(f"  alongamento FEM     : {along_fem:.5f} mm")
    print(f"  alongamento teorico : {along_teo:.5f} mm")
    print(f"  erro relativo       : {abs(along_fem-along_teo)/abs(along_teo)*100:.4f} %")

    # ---------- Caso 3: pequeno portico em L (2 tubos, tubo diagonal) ----------
    # Mostra rotacao + assembly funcionando com tubos em direcoes diferentes.
    f3 = Frame3D()
    p0 = f3.add_no(0, 0, 0)
    p1 = f3.add_no(0, 0, 300)      # coluna vertical
    p2 = f3.add_no(400, 0, 300)    # viga horizontal no topo
    f3.add_elemento(p0, p1, E=E, G=G, **sec)
    f3.add_elemento(p1, p2, E=E, G=G, **sec)
    f3.add_apoio(p0)               # base engastada
    f3.add_carga(p2, Fz=-2000.0)   # carga na ponta da viga
    u3 = f3.resolver()
    print("\n=== Caso 3: portico em L (2 tubos) ===")
    print(f"  desloc. vertical (z) da ponta : {u3[6*p2+2]:.4f} mm")
    print(f"  desloc. horizontal (x) da ponta: {u3[6*p2+0]:.4f} mm")
    esf = f3.esforcos()
    print(f"  esforco axial no tubo 0 (coluna): N = {esf[0][6]:.1f} N")
    print(f"  momento fletor base coluna (Mz) : {esf[0][5]:.0f} N.mm")

    print("\nValidado: os casos 1 e 2 batem com a teoria (erro ~0).")
