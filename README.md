# FEA-Simulator

Solver de elementos finitos para estruturas de barras tubulares (chassi tubular,
suspensão, treliça espacial), em Python. Modela cada tubo como elemento de viga 3D
(12 GDL), resolve `K · u = F` e recupera os esforços internos. Inclui visualização
3D interativa com PyVista.

## Arquivos

| Arquivo | Descrição |
|---|---|
| `frame3d.py` | Solver: montagem, resolução e recuperação de esforços |
| `visualizar.py` | Visualização 3D interativa |

## Instalação

```bash
pip install numpy pyvista
```

## Uso

```python
from frame3d import Frame3D, secao_tubo
from visualizar import visualizar_frame

E, G = 210_000.0, 80_000.0           # aço, em MPa (unidades N e mm)
sec = secao_tubo(D_ext=25.0, espessura=2.0)

f = Frame3D()
n0 = f.add_no(0, 0, 0)
n1 = f.add_no(500, 0, 0)
f.add_elemento(n0, n1, E=E, G=G, **sec)
f.add_apoio(n0)                       # engaste no nó 0
f.add_carga(n1, Fz=-1000.0)           # carga transversal no nó 1
f.resolver()

visualizar_frame(f, escala=30, modo='tensao')
```

Use unidades consistentes (o exemplo usa N e mm → E em MPa, A em mm², I/J em mm⁴).
Na visualização: `escala` exagera a deformada, `modo` pode ser `'tensao'` (axial),
`'vonmises'` ou `'simples'`. Cinza translúcido = forma original; cor = tensão;
cubos vermelhos = apoios.

<img width="1024" height="768" alt="image" src="https://github.com/user-attachments/assets/16df91f2-9e54-47df-9aea-adf3624afd94" />

## Como funciona

Método da rigidez direta: monta a matriz local 12×12 de cada tubo (axial, torção,
duas flexões), rotaciona para o referencial global, faz o assembly somando nos nós
compartilhados, aplica cargas e apoios, resolve para os deslocamentos e recupera os
esforços de cada tubo.

Validado contra soluções analíticas (viga em balanço, tração axial) com erro ~0% —
rode `python3 frame3d.py` para ver.

## Limitações

Cargas apenas nodais; vigas de Euler-Bernoulli (sem cisalhamento transversal);
análise linear estática (sem flambagem nem dinâmica); modo von Mises aproximado.
São extensões naturais sobre a mesma base.
