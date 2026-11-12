import matplotlib.pyplot as plt
import numpy as np
from config import AAR, PAAR, Hstart, Hend

def plotHistogram(hours, title):
    plt.figure()
    plt.hist(hours, bins=np.arange(0, 25, 1), edgecolor='black')
    plt.plot([0, Hstart], [AAR, AAR], 'g-', linewidth=3, label=f'AAR = {AAR}')
    plt.plot([Hend, 24], [AAR, AAR], 'g-', linewidth=3)
    plt.plot([Hstart, Hend], [PAAR, PAAR], 'r-', linewidth=3, label=f'PAAR = {PAAR}')
    plt.xlabel('Time [h]')
    plt.ylabel('Number of arrivals')
    plt.title(title)
    plt.xlim(0, 24)
    plt.grid(True)
    plt.legend(loc='upper left')

def choose_figures(figures, output_dir, interactive=True, choice=None):

    from pathlib import Path
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    names = list(figures)
    print('\nFigures (all are saved in the results folder):')
    for i, name in enumerate(names, 1):
        print(f' {i}. {name}')
    print(' a. Todas | n. Ninguna | Ejemplo: 1,3,5')
    if choice is None and interactive:
        try:
            choice = input('¿Qué gráficas quieres ver? [n]: ').strip() or 'n'
        except EOFError:
            choice = 'n'
    if choice is None:
        choice = 'n'
    choice = str(choice).strip().lower()
    if choice in ('a', 'all', 'todas'):
        selected = list(range(len(names)))
    elif choice in ('n', 'none', 'ninguna', ''):
        selected = []
    else:
        try:
            selected = sorted({int(s.strip())-1 for s in choice.split(',')})
            if any(i < 0 or i >= len(names) for i in selected):
                raise ValueError('Número fuera de rango')
        except ValueError:
            print('Selección no válida: no se abren gráficas.')
            selected = []
    for i, (name, fig) in enumerate(figures.items()):
        fig.savefig(folder / f'figure_{i+1}.png', dpi=150, bbox_inches='tight')  # all figures are saved
        if i not in selected:
            plt.close(fig)
    if selected:
        plt.show()
    return [names[i] for i in selected]
