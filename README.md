# AutoEncoder

## Introduction

This is a quick implementation based on Tensorflow2.x for auto-encoder-based signals anomaly detection.

The signals are generated through Matlab Simulink. Only the Vehicle speed signals are used in this work.
[Link to Matlab model](https://ww2.mathworks.cn/help/simulink/slref/modeling-an-automatic-transmission-controller.html;jsessionid=4834aa15a9bb5f0bdd260aa8a990?lang=en)

<p align="center">
    <img width="90%" src="/IMAGES/speed.png" style="max-width:100%;"></a>
</p>

## Quickstart

Quick signal anomaly detecting and visualizing the results

`python demo.py`

## Generate dataset

`python main_create_dataset.py`

## Train & test

To train a legacy mlp-based auto-encoder:

`python main_legacy_train`

To test a legacy mlp-based auto-encoder:

`python main_legacy_test`

To train and test a mlp-based auto-encoder with a trainable spike fault layer.

`python main_heuristic_train_search.py`

## Search for counter examples

Search for counter examples using hard search methods

`python main_hard_search.py`

Search for counter examples using heuristic search methods based on SGD optimization

`python main_heuristic_train_search.py`

## Parameters

The parameters are listed in `'./autoencoder/config.py'` and they are configurable.
