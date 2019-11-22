## GIF-7005: FORMAT SUGGÉRÉ POUR LES DONNÉES D'ENTRÉES ET LEURS ÉTIQUETTES

Tout ce qui peut être facilement récupérable en `numpy.array` serait à favoriser.

Même s'il est possible de les charger dans `python` avec le module `scipy.io`, j'éviterais de sauvegarder directement les données et leurs étiquettes en format `matlab` car leur portablilité est alors réduite.

Depuis `matlab`, vaudrait mieux les enregistrer en format texte séparé par virgule avec en-tête explicative et méta-données suffisantes. Un symbole couramment employé pour commenter les lignes d'en-tête est le dièse (`#`).

Le chargement dans `python` peut alors se faire directement avec `numpy` ou, encore mieux, en passant d'abord par `pandas`.

L'avantage de `pandas` c'est qu'on peut manipuler efficacement les données avant de les transformer en tableaux et vecteurs `numpy`. Les données manipulées peuvent aussi être ré-enregistrées en format binaire (`pickle`) directement compatibles avec `pandas`, ce qui est relativement plus rapide à recharger que des fichiers textes.

Une fois les données acquises en format `numpy`, leur transformation vers des tenseurs `PyTorch` est aisée.

Ci-dessous, je liste quelques références qui appuient ma suggestion.


#### SITE OFFICIEL DE [`PyTorch`](https://pytorch.org/)
En particulier, la documentation de développement pour le module [`torch.utils.data`](https://pytorch.org/docs/stable/data.html).


#### NOTES DU COURS DU 6 NOVEMBRE 2019

Document `PyTorch` page 16/25

```python
>>> import torch.utils.data
>>> help(torch.utils.data.Dataset)
>>> help(torch.utils.data.DataLoader)
```


#### MÉTHODE UNSQUEEZE

```python
>>> import torch
>>> help(torch.unsqueeze)
```

#### BLOGUE-NOTES SUR LES [PyTorch Tensor Data Types](https://jdhao.github.io/2017/11/15/pytorch-datatype-note/)


#### LIVRE DE [RECETTES](https://www.apress.com/gp/book/9781484242575)


___
pyfortin 20191121
##### EOF
