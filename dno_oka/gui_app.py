import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output, display

from evaluation import calculate_metrics
from gui_guard import (
    analysis_lock,
    build_root,
    ensure_widgets,
    get_widget,
    register_handlers_once,
    set_buttons_disabled,
    set_gui_context,
    set_results_tab,
)
from image_processing import get_overlay, preprocess_image, segment_vessels

METHODS = ['Filtr Frangi', 'Filtr Sato', 'Klasyfikator ML']

last_metrics = {}


def _status(html):
    get_widget('status_label').value = html


def _display_results(img_name):
    global last_metrics
    from gui_guard import _gui_context as ctx

    last_metrics = {}

    images_dir = ctx['images_dir']
    manual_dir = ctx['manual_dir']
    mask_dir = ctx['mask_dir']
    ml_segmenter = ctx['ml_segmenter']
    train_ml_model = ctx['train_ml_model']
    get_is_ml_trained = ctx['get_is_ml_trained']
    num_slider = get_widget('num_images_to_train')

    img_path = os.path.join(images_dir, img_name)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    base_name = os.path.splitext(img_name)[0]
    manual_mask = cv2.imread(
        os.path.join(manual_dir, base_name + '.tif'), cv2.IMREAD_GRAYSCALE
    )
    fov_mask = cv2.imread(
        os.path.join(mask_dir, base_name + '_mask.tif'), cv2.IMREAD_GRAYSCALE
    )

    preprocessed = preprocess_image(image_rgb)

    results = {}
    for method in METHODS:
        if method == 'Klasyfikator ML' and not get_is_ml_trained():
            _status(
                '<p>⏳ <b>Krok 1/2:</b> Trenowanie modelu ML na '
                f'{num_slider.value} zdjęciach... (może potrwać 1-2 min)</p>'
            )
            train_ml_model(num_slider.value)
        detected = segment_vessels(
            preprocessed,
            method=method,
            mask=fov_mask,
            ml_segmenter=ml_segmenter,
        )
        overlay = get_overlay(image_rgb, detected)
        metrics = calculate_metrics(manual_mask, detected, mask=fov_mask)
        results[method] = {
            'detected': detected,
            'overlay': overlay,
            'metrics': metrics,
        }
        last_metrics[method] = metrics

    output = get_widget('output')
    with output:
        clear_output(wait=True)

        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle(
            f'Wyniki segmentacji naczyń: {img_name}',
            fontsize=16,
            fontweight='bold',
        )

        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Oryginał', fontsize=13)
        axes[0, 0].axis('off')

        axes[0, 1].imshow(manual_mask, cmap='gray')
        axes[0, 1].set_title('Maska ekspercka (Ground Truth)', fontsize=13)
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')

        for i, method in enumerate(METHODS):
            axes[1, i].imshow(results[method]['detected'], cmap='gray')
            axes[1, i].set_title(f'{method}\n(wykryte naczynia)', fontsize=12)
            axes[1, i].axis('off')

        for i, method in enumerate(METHODS):
            m = results[method]['metrics']
            axes[2, i].imshow(results[method]['overlay'])
            axes[2, i].set_title(
                f'{method} - overlay\n'
                f'Acc: {m["accuracy"]:.4f}  Sens: {m["sensitivity"]:.4f}\n'
                f'Spec: {m["specificity"]:.4f}  G-Mean: {m["g_mean"]:.4f}',
                fontsize=10,
            )
            axes[2, i].axis('off')

        plt.tight_layout()
        plt.show()

        print(f'\n{"STATYSTYKI":^70}')
        print(
            f'{"Metoda":<20} | {"Accuracy":<10} | {"Sensitivity":<12} | '
            f'{"Specificity":<12} | {"G-Mean":<10}'
        )
        print('-' * 70)
        for method in METHODS:
            m = results[method]['metrics']
            print(
                f'{method:<20} | {m["accuracy"]:<10.4f} | {m["sensitivity"]:<12.4f} | '
                f'{m["specificity"]:<12.4f} | {m["g_mean"]:<10.4f}'
            )

    set_results_tab(0)
    _status(
        '<p style="color:green">✅ Analiza zakończona. '
        'Wyniki w zakładce <b>Wyniki analizy</b>.</p>'
    )


def execute_analysis():
    from gui_guard import _gui_context as ctx, _widgets

    if _widgets is None:
        raise RuntimeError('Najpierw uruchom komórkę GUI.')

    if not analysis_lock.acquire(blocking=False):
        return

    set_buttons_disabled(True)
    try:
        get_is_ml_trained = ctx['get_is_ml_trained']
        if not get_is_ml_trained():
            num = get_widget('num_images_to_train').value
            _status(
                '<p>⏳ <b>Krok 1/2:</b> Trenowanie modelu ML na '
                f'{num} zdjęciach...<br>'
                '<small>(pierwsze uruchomienie - może potrwać 1-2 minuty)</small></p>'
            )
        else:
            _status('<p>⏳ Przetwarzanie zdjęcia... proszę czekać</p>')

        with get_widget('cm_output'):
            clear_output()

        _display_results(get_widget('image_select').value)
    finally:
        set_buttons_disabled(False)
        analysis_lock.release()


def execute_show_confusion_matrices():
    if not last_metrics:
        _status(
            '<p style="color:orange">⚠️ Najpierw kliknij <b>Uruchom analizę</b>.</p>'
        )
        return

    cm_output = get_widget('cm_output')
    with cm_output:
        clear_output(wait=True)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Macierze pomyłek', fontsize=14, fontweight='bold')
        for i, method in enumerate(METHODS):
            cm = last_metrics[method]['confusion_matrix']
            im = axes[i].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            axes[i].set_title(f'{method}', fontsize=12)
            axes[i].set_xlabel('Predykcja')
            axes[i].set_ylabel('Prawda')
            tick_marks = np.arange(2)
            axes[i].set_xticks(tick_marks)
            axes[i].set_xticklabels(['Tło', 'Naczynia'])
            axes[i].set_yticks(tick_marks)
            axes[i].set_yticklabels(['Tło', 'Naczynia'])
            plt.colorbar(im, ax=axes[i])
            thresh = cm.max() / 2.0
            for r, c in np.ndindex(cm.shape):
                axes[i].text(
                    c,
                    r,
                    format(cm[r, c], 'd'),
                    ha='center',
                    va='center',
                    color='white' if cm[r, c] > thresh else 'black',
                )
        plt.tight_layout()
        plt.show()

    set_results_tab(1)
    _status(
        '<p>Macierze w zakładce <b>Macierze pomyłek</b>. '
        'Wróć do wykresów klikając <b>Wyniki analizy</b>.</p>'
    )


def execute_reset_ml():
    from gui_guard import _gui_context as ctx, _widgets

    if _widgets is None:
        raise RuntimeError('Najpierw uruchom komórkę GUI.')

    if not analysis_lock.acquire(blocking=False):
        return

    set_buttons_disabled(True)
    try:
        ctx['set_is_ml_trained'](False)
        _status('<p>⏳ Ponowne trenowanie modelu ML...</p>')
        with get_widget('output'):
            clear_output()
        with get_widget('cm_output'):
            clear_output()
        ctx['train_ml_model'](get_widget('num_images_to_train').value, force=True)
        set_results_tab(0)
        _status('<p style="color:green">✅ Model ML wytrenowany ponownie.</p>')
    finally:
        set_buttons_disabled(False)
        analysis_lock.release()


def _on_run_clicked(_btn):
    execute_analysis()


def _on_show_cm_clicked(_btn):
    execute_show_confusion_matrices()


def _on_reset_clicked(_btn):
    execute_reset_ml()


def display_gui(
    image_list,
    images_dir,
    manual_dir,
    mask_dir,
    ml_segmenter,
    train_ml_model,
    get_is_ml_trained,
    set_is_ml_trained,
):
    set_gui_context(
        images_dir=images_dir,
        manual_dir=manual_dir,
        mask_dir=mask_dir,
        ml_segmenter=ml_segmenter,
        train_ml_model=train_ml_model,
        get_is_ml_trained=get_is_ml_trained,
        set_is_ml_trained=set_is_ml_trained,
    )

    ensure_widgets(image_list)
    register_handlers_once(_on_run_clicked, _on_show_cm_clicked, _on_reset_clicked)

    clear_output(wait=True)
    display(build_root())


def get_num_images_slider():
    from gui_guard import _widgets
    if _widgets is None:
        return None
    return _widgets['num_images_to_train']
