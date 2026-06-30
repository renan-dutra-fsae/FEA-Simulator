"""
visualizar.py — Visualizacao 3D de estruturas de barras com PyVista.

Conecta direto ao solver Frame3D (frame3d.py):
  - desenha a estrutura indeformada (cinza, translucida)
  - desenha a deformada (colorida por tensao), com escala ajustavel
  - colore cada tubo pela tensao axial (ou von Mises simplificada)
  - camera interativa: orbitar, zoom, pan com o mouse

Uso tipico (depois de resolver o modelo):
    from frame3d import Frame3D, secao_tubo
    from visualizar import visualizar_frame
    ...
    f.resolver()
    visualizar_frame(f, escala=50, modo='tensao')

Para salvar uma imagem em vez de abrir janela (util em servidor/notebook):
    visualizar_frame(f, escala=50, screenshot='chassi.png')
"""

import numpy as np
import pyvista as pv


def _tensao_axial_por_elemento(frame):
    """
    Calcula a tensao axial (N/A) em cada tubo a partir dos esforcos.
    Retorna array com uma tensao por elemento (MPa, se unidades N e mm).
    Usa o esforco normal no no j (indice 6 do vetor de esforcos locais).
    """
    esforcos = frame.esforcos()
    tensoes = []
    for el, f_loc in zip(frame.elementos, esforcos):
        N = f_loc[6]              # esforco normal (tracao>0)
        A = el['A']
        tensoes.append(N / A)     # MPa
    return np.array(tensoes)


def _tensao_von_mises_por_elemento(frame):
    """
    Estimativa de von Mises combinando axial + flexao + torcao na fibra
    extrema de cada tubo. Aproximacao para colorir/ranquear tubos criticos.
    Assume tubo circular: usa Iy=Iz=I, raio externo c estimado de A e I.
    """
    esforcos = frame.esforcos()
    vms = []
    for el, f_loc in zip(frame.elementos, esforcos):
        N = f_loc[6]
        My = max(abs(f_loc[4]), abs(f_loc[10]))   # momento fletor (pior no)
        Mz = max(abs(f_loc[5]), abs(f_loc[11]))
        Tq = abs(f_loc[9])                          # torcao
        A, I, J = el['A'], el['Iy'], el['J']
        # raio externo estimado a partir de I e A (tubo): aproximacao
        c = np.sqrt(2 * I / A + (A / (4 * np.pi)))  # estimativa grosseira de c
        M = np.sqrt(My**2 + Mz**2)
        sigma = abs(N) / A + M * c / I              # normal (axial + flexao)
        tau = Tq * c / J                            # cisalhamento torcional
        vms.append(np.sqrt(sigma**2 + 3 * tau**2))  # von Mises
    return np.array(vms)


def _polydata_da_estrutura(coords, elementos):
    """
    Monta um PolyData do PyVista com os nos como pontos e os tubos
    como celulas de linha (cada tubo = 2 pontos).
    """
    pts = np.array(coords, float)
    # formato de linhas do VTK: [2, i, j, 2, i, j, ...] (2 = pontos por linha)
    linhas = []
    for el in elementos:
        linhas += [2, el['no_i'], el['no_j']]
    poly = pv.PolyData()
    poly.points = pts
    poly.lines = np.array(linhas)
    return poly


def visualizar_frame(frame, escala=1.0, modo='tensao',
                     raio_tubo=None, screenshot=None,
                     mostrar_indeformada=True):
    """
    Renderiza a estrutura resolvida em 3D.

    frame  : objeto Frame3D ja resolvido (.resolver() chamado)
    escala : fator de exagero da deformada (ex.: 50 => desloc x50)
    modo   : 'tensao' (axial N/A), 'vonmises', ou 'simples' (cor unica)
    raio_tubo : raio visual dos tubos; se None, estima automaticamente
    screenshot : se dado um caminho, salva PNG em vez de abrir janela
    mostrar_indeformada : desenha a estrutura original translucida
    """
    coords = np.array(frame.nos, float)
    n_nos = len(coords)

    # ---- deslocamentos translacionais de cada no (3 primeiros de cada 6) ----
    desloc = np.zeros((n_nos, 3))
    for i in range(n_nos):
        desloc[i] = frame.u[6*i:6*i+3]

    coords_def = coords + desloc * escala

    # ---- raio visual automatico ----
    if raio_tubo is None:
        bbox = coords.max(axis=0) - coords.min(axis=0)
        diag = np.linalg.norm(bbox)
        raio_tubo = diag * 0.006 + 1e-9

    # ---- escalar por elemento (cor) ----
    if modo == 'tensao':
        valores = _tensao_axial_por_elemento(frame)
        titulo_barra = 'Tensao axial (MPa)'
    elif modo == 'vonmises':
        valores = _tensao_von_mises_por_elemento(frame)
        titulo_barra = 'von Mises (MPa)'
    else:
        valores = None

    plotter = pv.Plotter()
    plotter.set_background('white')

    # ---- estrutura indeformada (referencia translucida) ----
    if mostrar_indeformada:
        poly0 = _polydata_da_estrutura(coords, frame.elementos)
        tubos0 = poly0.tube(radius=raio_tubo * 0.6)
        plotter.add_mesh(tubos0, color='lightgray', opacity=0.35)

    # ---- deformada ----
    poly1 = _polydata_da_estrutura(coords_def, frame.elementos)
    if valores is not None:
        # associa um valor escalar a cada celula (tubo)
        poly1.cell_data['valor'] = valores
        tubos1 = poly1.tube(radius=raio_tubo)
        plotter.add_mesh(tubos1, scalars='valor', cmap='turbo',
                         scalar_bar_args={'title': titulo_barra})
    else:
        tubos1 = poly1.tube(radius=raio_tubo)
        plotter.add_mesh(tubos1, color='steelblue')

    # ---- nos como esferas ----
    plotter.add_mesh(pv.PolyData(coords_def),
                     color='black', point_size=8, render_points_as_spheres=True)

    # ---- apoios marcados (cubos vermelhos) ----
    for no in frame.apoios:
        plotter.add_mesh(
            pv.Cube(center=coords_def[no], x_length=raio_tubo*4,
                    y_length=raio_tubo*4, z_length=raio_tubo*4),
            color='red')

    plotter.add_axes()
    plotter.add_text(f'Deformada (escala x{escala:g})', font_size=10)

    if screenshot:
        plotter.show(screenshot=screenshot)
        print(f'Imagem salva em: {screenshot}')
    else:
        plotter.show()
    return plotter


# ----------------------------------------------------------------------
# DEMO: gera um pequeno chassi 3D, resolve e renderiza para PNG
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    from frame3d import Frame3D, secao_tubo

    pv.OFF_SCREEN = True   # modo sem janela (servidor)

    E, G = 210_000.0, 80_000.0
    sec = secao_tubo(D_ext=25.0, espessura=2.0)

    # mini-chassi: um "cubo" aberto de tubos (8 nos, varias barras)
    f = Frame3D()
    P = [
        (0, 0, 0), (400, 0, 0), (400, 300, 0), (0, 300, 0),       # base
        (0, 0, 350), (400, 0, 350), (400, 300, 350), (0, 300, 350) # topo
    ]
    ns = [f.add_no(*p) for p in P]

    arestas = [
        (0,1),(1,2),(2,3),(3,0),          # base
        (4,5),(5,6),(6,7),(7,4),          # topo
        (0,4),(1,5),(2,6),(3,7),          # montantes
        (0,5),(1,6),(3,4)                 # diagonais (rigidez)
    ]
    for i, j in arestas:
        f.add_elemento(ns[i], ns[j], E=E, G=G, **sec)

    # base engastada nos 4 nos inferiores
    for k in range(4):
        f.add_apoio(ns[k])

    # carga lateral no topo (simula esforco dinamico)
    f.add_carga(ns[5], Fx=3000.0, Fz=-1500.0)
    f.add_carga(ns[6], Fx=3000.0)

    f.resolver()

    # tensao axial maxima (em modulo) para referencia
    tens = _tensao_axial_por_elemento(f)
    print(f"Tubos: {len(f.elementos)} | tensao axial max: {np.abs(tens).max():.1f} MPa")
    print(f"Desloc. max (no): {np.abs([f.u[6*i:6*i+3] for i in range(len(f.nos))]).max():.3f} mm")

    visualizar_frame(f, escala=30, modo='tensao', screenshot='chassi_demo.png')
