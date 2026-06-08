"""Singleton widgetów — przyciski rejestrowane tylko raz na sesję kernela."""
import threading

import ipywidgets as widgets

analysis_lock = threading.Lock()
training_lock = threading.Lock()
holdout_lock = threading.Lock()

_gui_context = {}
_widgets = None
_gui_root = None
_handlers_registered = False


def set_gui_context(**kwargs):
    _gui_context.update(kwargs)


def get_widget(name):
    return _widgets[name]


def ensure_widgets(image_list):
    global _widgets
    if _widgets is not None:
        _widgets['image_select'].options = image_list
        _widgets['num_images_to_train'].max = max(1, len(image_list) - 1)
        return _widgets

    max_train = max(1, len(image_list) - 1)
    _widgets = {
        'image_select': widgets.Dropdown(
            options=image_list,
            description='Zdjęcie:',
        ),
        'num_images_to_train': widgets.IntSlider(
            value=min(3, max_train),
            min=1,
            max=max_train,
            step=1,
            description='Zdj. do treningu ML:',
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d',
            style={'description_width': 'initial'},
        ),
        'run_button': widgets.Button(
            description='Uruchom analizę',
            button_style='success',
            icon='play',
            layout=widgets.Layout(width='180px'),
        ),
        'show_cm_button': widgets.Button(
            description='Pokaż macierze pomyłek',
            button_style='info',
            icon='table',
            layout=widgets.Layout(width='220px'),
        ),
        'reset_ml_button': widgets.Button(
            description='Resetuj ML i CNN',
            button_style='warning',
            icon='refresh',
            layout=widgets.Layout(width='200px'),
        ),
        'status_label': widgets.HTML(value=''),
        'output': widgets.Output(),
        'cm_output': widgets.Output(),
        'results_tabs': widgets.Tab(),
    }
    _widgets['results_tabs'].children = [_widgets['output'], _widgets['cm_output']]
    _widgets['results_tabs'].set_title(0, 'Wyniki analizy')
    _widgets['results_tabs'].set_title(1, 'Macierze pomyłek')
    return _widgets


def register_handlers_once(run_cb, show_cm_cb, reset_cb):
    """Handlery podpinane dokładnie raz — kolejne Run All tylko aktualizuje kontekst."""
    global _handlers_registered
    if _handlers_registered:
        return

    for btn in (
        _widgets['run_button'],
        _widgets['show_cm_button'],
        _widgets['reset_ml_button'],
    ):
        btn._click_handlers.callbacks.clear()

    _widgets['run_button'].on_click(run_cb)
    _widgets['show_cm_button'].on_click(show_cm_cb)
    _widgets['reset_ml_button'].on_click(reset_cb)
    _handlers_registered = True


def set_results_tab(index):
    _widgets['results_tabs'].selected_index = index


def set_buttons_disabled(disabled):
    _widgets['run_button'].disabled = disabled
    _widgets['show_cm_button'].disabled = disabled
    _widgets['reset_ml_button'].disabled = disabled


def build_root():
    global _gui_root
    _gui_root = widgets.VBox([
        widgets.HBox([_widgets['image_select'], _widgets['num_images_to_train']]),
        widgets.HBox([
            _widgets['run_button'],
            _widgets['show_cm_button'],
            _widgets['reset_ml_button'],
        ]),
        _widgets['status_label'],
        _widgets['results_tabs'],
    ])
    return _gui_root
