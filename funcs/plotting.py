import seaborn as sns
from . import database as db
import pandas as pd
from matplotlib import pyplot as plt


def change_axis_legend_cols(axis, ncol):
    # https://stackoverflow.com/a/56576849
    h_, l_ = axis.get_legend_handles_labels()
    axis.legend_.remove()
    axis.legend(h_, l_, ncol=ncol)


def subscript(
        s,
        reason='plot'
):
    s2 = s[0]
    unicode_dict = {
        '0': '\u2080',
        '1': '\u2081',
        '2': '\u2082',
        '3': '\u2083',
        '4': '\u2084',
        '5': '\u2085',
        '6': '\u2086',
        '7': '\u2087',
        '8': '\u2088',
        '9': '\u2089',
    }
    for i in range(1, len(s)):
        if s[i-1] != ' ' and s[i].isdigit():
            if reason.lower() in {'str', 'string', 's'}:
                s2 += unicode_dict[s[i]]
            elif reason.lower in {'plt', 'plot', 'p'}:
                s2 += '$_' + s[i] + '$'
            else:
                s2 += s[i]
        else:
            s2 += s[i]
    return s2


def sensitivity_plot(
        database,
        table_id,
        sensitivity_type='cell_size',
        calc_method='Gavrikov',
        min_threshold=0,
        display_top=0
):
    methods_str = {
        'Gavrikov': '_gav',
        'gav': '_gav',
        'g': '_gav',
        'Westbrook': '_west',
        'west': '_west',
        'w': '_west',
        'Ng': '_ng',
        'ng': '_ng',
        'n': '_ng'
    }
    methods_title = {
        'Gavrikov': '(Gavrikov Method)',
        'gav': '(Gavrikov Method)',
        'g': '(Gavrikov Method)',
        'Westbrook': '(Westbrook Method)',
        'west': '(Westbrook Method)',
        'w': '(Westbrook Method)',
        'Ng': '(Ng Method)',
        'ng': '(Ng Method)',
        'n': '(Ng Method)'
    }
    sens_str = {
        'cell_size': 'cell_size',
        'cell size': 'cell_size',
        'c': 'cell_size',
        'ind_len': 'ind_len',
        'induction length': 'ind_len',
        'i': 'ind_len'
    }
    sens_title = {
        'cell_size': 'Cell Size ',
        'cell size': 'Cell Size ',
        'c': 'Cell Size ',
        'ind_len': 'Induction Length ',
        'induction length': 'Induction Length ',
        'i': 'Induction Length '
    }
    table_name = 'data'

    # gather and sort information
    table = db.Table(
        database,
        table_name
    )
    imported_data = table.fetch_pert_table(
        rxn_table_id=table_id
    )
    df = pd.DataFrame(imported_data)
    sort_key = 'sens_' + sens_str[sensitivity_type] + methods_str[calc_method]
    sorted_df = df.iloc[df[sort_key].abs().argsort()]
    sorted_df = sorted_df[sorted_df[sort_key].abs() > min_threshold]
    if display_top > 0:
        sorted_df = sorted_df[-display_top:]

    # replace arrows and switch to subscripts
    sorted_df.rxn = sorted_df.rxn.\
        str.replace('<=>', '\u2194')\
        .str.replace('=>', '\u2192')\
        .str.replace('<=', '\u2190')\
        .apply(subscript, args=('s',))

    sns.set(style='whitegrid')
    sns.set_color_codes("deep")
    sns.set_context('paper')
    sns.set_style({'font.family': 'serif', 'font.serif': 'Computer Modern'})
    fig, ax = plt.subplots(figsize=(8, 3))
    sns.barplot(
        ax=ax,
        x=sort_key,
        y='rxn',
        data=sorted_df,
        label='Westbrook',
        color='b'
    )
    lbl_font = {
        'weight': 'bold',
        'size': 12
    }
    title_font = {
        'weight': 'bold',
        'size': 18
    }
    plt.xlabel('Normalized Sensitivity Coefficient', fontdict=lbl_font)
    plt.ylabel('Reaction', fontdict=lbl_font)
    plt.title(
        sens_title[sensitivity_type] +
        'Sensitivity\n' +
        methods_title[calc_method],
        fontdict=title_font
    )

    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    return fig
