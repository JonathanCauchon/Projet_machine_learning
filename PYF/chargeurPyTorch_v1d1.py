#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
\file chargeurPyTorch.py


\author pyfortin


\date 2019.12.04


\version 1.0 (celle présentée à l'équipe le 2 Décembre)
\version 1.1 (modifiée pour le 4 Décembre)


\brief
Paréparateur de données brutes pour chargement PyTorch


\details
Sera amélioré selon feedbacks de l'équipe.


\todo
TODO: implémenter la combinaison des parties réelle et imaginaire
de E_drop ou E_thru pour produire un nouvel attribut d'amplitude
et/ou un nouvel attribut de phase.

\todo
TODO: Vérifier la soi-disant "compression" que Hammond [3] effectue
au Chapitre 3, en particulier à la section 3.2.1. Noter que
la Figure 4.3 illustre exactement ce qu'on tente de faire ici.

\todo
TODO: normaliser balises doxygen et générer documentation.

\note
{
	Description des fichiers de données
	===================================

	.en concaténant tous les fichiers de données on totalise nbTotalDonnees
	.chaque fichier de données contient

		nbDonneesJC = selon argument de ChirpedContraDC.createDataSet
		nbDonneesJSY = selon batchSize de generateDataForNN_loop.m
		nbLignes = 1+nbDonnees
		nbAttributsJC = 9
		nbAttributsJSY = 8

	.la 1ere ligne décrit les nbAttributs séparés par des virgules
	.les nbDonnees* lignes suivantes contiennent les valeurs des attributs
	.certains attributs sont en plusieurs dimensions
	.les attributs multi-dimensionnels sont enregistrés sur plus d'une colonne
	.pour chaque donnée, les valeurs d'attributs sont séparées par des espaces

	utilisons des indices d'attributs débutant à 0
	utilisont des indices de colonnes débutant à 0

	ainsi, le décalage en colonnes de la première -- ou seule -- valeur
	numérique rattachée à chaque attribut sera donnée par

		$\delta_i = \sum_{j=0}^{i-1}{d_\kappa(j)}$

	(a): colonne $i$ de la 1ere ligne de donnée brute (en-tête)
	(b): idAttribut de cette colonne $\kappa(i)$
	(c): symbole texte $\sigma_{\kappa(i)}$ relié à cet attribut
	(d): type $\tau_{\kappa(i)}$ relié à cet attribut
	(e): domaine $\rau_{\kappa(i)}$ relié à cet attribut
	(f): dimensions $d_{\kappa(i)}$ relié à cet attribut
	(g): décalage $\delta_i$ en nombre de colonnes pour les lignes de valeurs

	Données JC
	----------------------------------------------------------------------
	(a) (b) (c)            (d)        (e)       (f)tot=4209   (g)
	----------------------------------------------------------------------
	0   0   a              float      R+        1             0
	1   1   N              int        N*        1             1
	2   2   kappa          float      R+*       1             2
	3   4   apodization    float      R+        101           3
	4   5   period         float      R+*       101           104
	5   6   real(E_drop)   float      R         1001          205
	6   7   imag(E_drop)   float      R         1001          1206
	7   8   real(E_thru)   float      R         1001          2207
	8   9   imag(E_thru)   float      R         1001          3208
	----------------------------------------------------------------------

	Données JSY
	----------------------------------------------------------------------
	(a) (b) (c)            (d)        (e)       (f)tot=2208   (g)
	----------------------------------------------------------------------
	0   0   a              float      R+        1             0
	1   1   N              int        N*        1             1
	2   2   kappa          float      R+*       1             2
	3   3   lambdaB        float      R+*       1             3
	4   4   apodization    float      R+        101           4
	5   5   period         float      R+*       101           105
	6   6   real(E_drop)   float      R         1001          206
	7   7   imag(E_drop)   float      R         1001          1207
	----------------------------------------------------------------------

	On distingue l'ensemble des réels R de l'ensemble R+ des réels positifs
	incluant zéro et de l'ensemble R+* des réels exclusivement positifs.

	On distingue l'ensemble des naturels N incluant le zéro de
	l'ensemble N* des naturels exclusivement positifs.
}


\note
{
	D'après la partie "III. Jeu de données" rédigée par Jonathan St-Yves
	dans la proposition de projet, il conviendrait d'effectuer le
	remodellage des données d'entrée pour en faire un échantillonnage
	de t = 1..nbTotalDonnees dont les étiquettes sont

		$$
		r^t = [ a , N , kappa , lambdaB ]
		$$

	et dont les données sont soit basées sur le champ électrique
	mesuré en sortie du port *drop*

		$$
		\mathcal{X}_E = ( x_E^t, r^t )
		$$

	avec

		$$
		x_E^t = [ real(E_drop) , imag(E_drop) ]
		||x_E^t|| = 2*(1+NB_LAMBDAS) \text{<--- cardinalité}
		$$

	soit basées sur les paramètres géométriques du réseau

		$$
		\mathcal{X}_G = ( x_G^t, r^t )
		$$

	avec

		$$
		x_G^t = [ apodization , period ]
		||x_G^t|| = 2*(1+NB_BLOCS) \text{<--- cardinalité}
		$$

	Peut être sera-t-il nécessaire de combiner les deux selon

		$$
		\mathcal{X}$ = ( x^t, r^t )
		$$

	avec

		$$
		x^t = [ apodization , period , real(E_drop) , imag(E_drop) ]
		$$

	Un échantillonnage possible serait aussi E_drop complexe
	Ou bien peut-être vaudra-t-il mieux séparer complètement la
	partie réelle de la partie imaginaire...
}


\warning
{
	Lorsque `type(X) == complex` alors `torch.from_numpy(X)` chiâle:

	\verbatim
	TypeError: can't convert numpy.ndarray of type numpy.complex128. The only
	supported types are: float{64|32|16}, int{64|32|16|8}, uint8, and bool.
	\endverbatim

	Une solution serait peut-être d'utiliser des modules externes
	tels que

	<a href="http://wavefrontshaping.net/index.php/component
	/content/article/69-community/tutorials/others
	/157-complex-valued-networks-with-pytorch-for-physics-applications">
	`complexPyTorch`</a>

	ou bien

	<a href="https://github.com
	/williamFalcon/pytorch-complex-tensor">`pytorch-complex-tensor`</a>.
}
"""

r"""
references.bib[debut]----------------------------------------------------------
% ici, lien naïf [1]
@mastersthesis{ st2017contra,
  title={Contra-directional couplers as optical filters
		   on the silicon on insulator platform},
  author={St-Yves, Jonathan},
  year={2017},
  school={Université Laval},
}


% ici, lien naïf [2]
@article{ shi2013silicon,
  title={Silicon photonic grating-assisted, contra-directional couplers},
  author={Shi, Wei
		    and Wang, Xu and Lin, Charlie and Yun, Han
		    and Liu, Yang and Baehr-Jones, Tom and Hochberg, Michael
			and Jaeger, Nicolas A. F. and Chrostowski, Lukas},
  journal={Optics Express},
  volume={21},
  number={3},
  pages={3633--3650},
  year={2013},
  publisher={Optical Society of America}

% ici, lien naïf [3]
@mastersthesis{ hammond2019machine,
  title={Machine Learning Methods for Nanophotonic
		   Design, Simulation, and Operation},
  author={Hammond, Alec Michael},
  year={2019},
  school={Brigham Young University},
}
references.bib[fin]------------------------------------------------------------
"""


# ------------------------- # -------------------------------------------------
# imports                   # méthodes et sous-modules et utilisés
# ------------------------- # -------------------------------------------------

import pandas               # read_csv

import pickle               # dump, load, HIGHEST_PROTOCOL

import numpy                # arange, array, reshape, ndim, append,
                            # all, sort, astype, random.randint, true_divide

import torch                # from_numpy, save, load, eq, all,

import torch.utils.data     # Dataset, DataLoader

import matplotlib           # figure.Figure.suptitle
					        # axes.Axes.{set_title,set_axis_off,scatter}

import matplotlib.pyplot    # subplots

import os                   # listdir, makedirs,
                            # path.{normpath, exists, join}

import sys                  # exit

import argparse             # ArgumentParser, RawDescriptionHelpFormatter

import textwrap             # indent



r"""
\var DIR_JC

\brief
Répertoire contenant données brutes générées par
code python de Jonathan Cauchon.
"""
#DIR_JC = '../jc/donnees' # ensemble complet et volumineux
DIR_JC = '../jc/d.petit' # sous-ensemble léger pour essais et débogage


r"""
\var EXT_JC

\brief
Extention fichiers de données brutes générées par
code python de Jonathan Cauchon.
"""
EXT_JC = '.txt'


r"""
\var ATTRIBUTS_JC

\brief
Identifiants des attributs des données brutes générées par
code python de Jonathan Cauchon.
"""
ATTRIBUTS_JC = [0,1,2,4,5,6,7,8,9]



r"""
\var DIR_JSY

\brief
Répertoire contenant données brutes générées par
code matlab de Jonathan St-Yves.
"""
#DIR_JSY = '../jsy/donnees' # ensemble complet et volumineux
DIR_JSY = '../jsy/d.petit' # sous-ensemble léger pour essais et débogage


r"""
\var EXT_JSY

\brief
Extention fichiers de données brutes générées par
code matlab de Jonathan St-Yves.
"""
EXT_JSY = '.txt'


r"""
\var ATTRIBUTS_JSY

\brief
Identifiants des attributs des données brutes générées par
code matlab de Jonathan St-Yves.
"""
ATTRIBUTS_JSY = [0,1,2,3,4,5,6,7]


r"""
\var DIR_STRUCT

\brief
Répertoire d'enregistrement des données structurées par ce script.
"""
DIR_STRUCT = '../struct'


r"""
\var NOM_PICKLE

\brief
Nom du fichier Pickle contenant toutes les données structurées.
"""
NOM_PICKLE = 'dStruct.pickle'


r"""
\var ATTRIBUTS_X

\brief
Attributs par défaut choisis pour constituer les données d'entrée.
"""
ATTRIBUTS_X = set([0,1,2])


r"""
\var ATTRIBUTS_y

\brief
Attributs par défaut reliés aux données d'entrée.
"""
ATTRIBUTS_y = set([6,7])

r"""
\var NOM_PYTORCH_X

\brief
Nom du fichier pytorch contenant les donnees de l'échantillonnage.
"""
NOM_PYTORCH_X = 'donnees.pt'


r"""
\var NOM_PYTORCH_y

\brief
Nom du fichier pytorch contenant les étiquettes de l'échantillonnage.
"""
NOM_PYTORCH_y = 'etiquettes.pt'


r"""
\var DOMAINES

\brief
Domaines possibles pour l'appartenance des valeurs numériques des attributs.
"""
DOMAINES = ['REEL','REELPOS','REELPOS*','NAT','NAT*']


r"""
\var NB_BLOCS

\brief
Nombre de segments égaux dans lesquels sont analysés le réseau.

\details
Nombre de blocs successifs de longueurs physiques (m) égales dans
lesquels on décompose le réseau afin d'en effectuer l'analyse par méthode
de matrice de transfert. Comme les positions des extrémités inférieures
et supérieures de l'union de ces blocs seront aussi considérées, il y
aura donc 1+NB_BLOCS positions dans les profils.
"""
NB_BLOCS = 100


r"""
\var NB_LAMBDAS

\brief
Nombre de segments égaux de 1 nm dans le domaine des longueurs d'ondes.

\details
Nombre d'intervalles de longueurs d'ondes (m) dans lesquels les spectres
de sortie sont discrétisés. On suppose ces intervalles successifs mesurent
tous $10^{-9}$ (m). Commes les extrémités inférieures et supérieures de
l'union de ces intervalles seront aussi considérées, il y aura
donc 1+NB_LAMBDAS longueurs d'ondes disctinctes dans les spectres.
"""
NB_LAMBDAS = 1000


r"""
\var dicoDesAttributs

\brief
Dictionnaire aidant au chargement des données étiquettées.

\note
La maintenance de l'ordre des attributs est assurée par
l'emploi de clés numériques entières débutant par zéro.

\note
Permet d'afficher la documentation avec une interface gui.

\note
Permet de facilement adapter le processus de chargement à
une modification du format des données brutes.

\warning
Les unités et les descriptions doivent être vérifiées par les experts.
"""
# FIXME, TODO, ??? : svp vérifier les unités
# FIXME, TODO, ??? : svp vérifier les descriptions des attributs
dicoDesAttributs = {
		0:{
			 'symbole':'a',
			 'unites':'1/(m.m)',
			 'description':
				  "Coefficient de l'exposant de la gaussienne qui "
				  "module le profil d'accentuation des "
				  "corrugations causant le couplage; d'après [1] section 2.6 "
				  "son expression semble être : $\exp(-a*(z)^2)$, où $z=0$ "
				  "serait le plan de coupe central perpendiculaire "
				  "à l'axe du réseau.",
			 'domaine':'REELPOS',
			 'dimensions': 1,
			 'usage':'X (ou y)'
		  },
		1:{
			 'symbole':'N',
			 'unites':'-',
			 'description':
				 "Nombre total de périodes du réseau; chacune "
				 "de ces périodes (ou franges) coincide également avec "
				 "une période de corrugation (perturbation) permettant "
				 "de générer du couplage.",
			 'domaine':'NAT*',
			 'dimensions': 1,
			 'usage':'X (ou y)'
		  },
		2:{
			 'symbole':'kappa',
			 'unites':'1/m',
			 'description':
				 "Valeur maximale de l'enveloppe de la gaussienne qui "
				 "module le profil d'accentuation des "
				 "corrugations causant le couplage; cette valeur "
				 "maximale survient au milieu du réseau.",
			 'domaine':'REELPOS*',
			 'dimensions': 1,
			 'usage':'X (ou y)'
		  },
		3:{
			 'symbole':'lambdaB',
			 'unites':'m',
			 'description':
				 "Longueur d'onde centrale du port *drop* dans le "
				 "coupleur contra-directionnel; correspond à $\lambda_D$ "
				 "dans l'article de Wei Shi [2] et le mémoire de "
				 "Jonathan St-Yves [1].",
			 'domaine':'REELPOS*',
			 'dimensions': 1,
			 'usage':'X (ou y)'
		  },
		4:{
			 'symbole':'apodization',
			 'unites':'1/m',
			 'description':
				  "Valeur de l'enveloppe d'apodisation "
				  "discrétisée à chaque extrémité des blocs d'analyse "
				  "employés par la méthode des matrices de transfert",
			 'domaine':'REELPOS',
			 'dimensions': 1+NB_BLOCS,
			 'usage':'X (ou y)'
		  },
		5:{
			 'symbole':'period',
			 'unites':'1/m',
			 'description':
				 "Période [moyenne?] des franges du réseau "
				 "discrétisée à chaque extrémité des blocs d'analyse "
				 "employés par la méthode des matrices de transfert",
			 'domaine':'REELPOS*',
			 'dimensions': 1+NB_BLOCS,
			 'usage':'X (ou y)'
		  },
		6:{
			 'symbole':'real(E_drop)',
			 'unites':'V/m',
			 'description':
				  "Partie réelle "
				  "du champ électrique mesuré au port *drop* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
			 'usage':'y (ou X) [~1]'
		  },
		7:{
			 'symbole':'imag(E_drop)',
			 'unites':'V/m',
			 'description':
				  "Partie imaginaire "
				  "du champ électrique mesuré au port *drop* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
			 'usage':'y (ou X) [~1]'
		  },
		8:{
			 'symbole':'real(E_thru)',
			 'unites':'V/m',
			 'description':
				  "Partie réelle "
				  "du champ électrique mesuré au port *thru* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
			 'usage':'y (ou X) [~2]'
		  },
		9:{
			 'symbole':'imag(E_thru)',
			 'unites':'V/m',
			 'description':
				  "Partie imaginaire "
				  "du champ électrique mesuré au port *thru* "
				  "du coupleur contra-directionnel.",
			 'domaine':'REEL',
			 'dimensions': 1+NB_LAMBDAS,
			 'usage':'y (ou X) [~2]'
		  },
}


def domaineValide(valeur,domaine):
	r"""
	\brief
	Valide l'appartenance d'une valeur à un domaine donné.

	\param[in] valeur
	Valeur numérique dont on veut vérifier l'appartenance à un domaine;
	les glyphes de la valeur peuvent être contenues dans une chaîne de
	caractères, mais la chaîne doit alors être convertible en valeur
	numérique.

	\param[in] domaine
	Domaine numérique auquel la valeur doit appartenir;
	doit faire partie d'une liste pré-définie de domaines.

	\return
	Vrai seulement si valeur appartient au domaine; faux sinon.

	\note
	Pourrait être amélioré pour vérifier la proximité avec zéro des
	nombres de faible grandeur en considérant l'epsilon machine.
	Mais cela nécessiterait alors de fixer la précision des nombres
	à virgules flottantes manipulés. Par exemple, pour
	une machine donnée:
		numpy.finfo(numpy.float32).eps = 1.1920929e-07
		numpy.finfo(numpy.float64).eps = 2.220446049250313e-16
		numpy.finfo(numpy.float128).eps = 1.084202172485504434e-19

	\warning
	Hormis les builtin types, seul les "numpy.int64" sont attendus
	commme instance de types pour les valeurs.
	"""
	try:
		assert(domaine in DOMAINES)
		if type(valeur) == str:
			assert(valeur.isnumeric())
			valeur = float(valeur)
		if domaine == 'NAT*':
			if isinstance(valeur, float):
				assert(valeur%1 == 0)
				assert(valeur>0)
			elif isinstance(valeur, int):
				assert(valeur>0)
			elif isinstance(valeur, numpy.int64):
				assert(valeur>0)
			else:
				raise
		if domaine == 'NAT':
			if isinstance(valeur, float):
				assert(valeur%1 == 0)
				assert(valeur>=0)
			elif isinstance(valeur, int):
				assert(valeur>=0)
			elif isinstance(valeur, numpy.int64):
				assert(valeur>=0)
			else:
				raise
		if domaine == 'REELPOS*':
			assert(isinstance(valeur, float))
			assert(valeur>0)
		if domaine == 'REELPOS':
			assert(isinstance(valeur, float))
			assert(valeur>=0)
		if domaine == 'REEL':
			assert(isinstance(valeur, float))
	except:
		print("!!! Erreur de domaineValide "
			"(valeur,domaine) = (%s,%s)"
			%(valeur,domaine))
		return False
	return True


def decalageEnColonnes(idAttribut,listeIdAttributs):
	r"""
	\brief
	Calcule le décalage en colonnes de la valeur d'un attribut.

	\param[in] idAttribut
	Clé numérique entière identifiant l'attribut.

	\param[in] listeIdAttributs
	Liste ordonnée des identifiants d'attributs présents dans
	le fichier de données.

	\return
	Décalage en colonnes de la première (ou seule) valeur numérique
	appartenant à l'attribut demandé dans les fichiers de données brutes.
	None si idAttribut ne correspond pas à un identifiant numérique valide.

	\note
	La première colonne, celle la plus à gauche sur une rangée donnée,
	est numérotée 0; la seconde est numérotée 1, ...
	"""
	try:
		assert(domaineValide(idAttribut,'NAT'))
		assert(idAttribut in listeIdAttributs)
		assert(idAttribut in dicoDesAttributs.keys())
	except:
		print("!!! Erreur de calcul de décalage "
			"(idAttribut,listeDesAttributs) = (%s,%s)"
			%(idAttribut,listeIdAttributs))
		return None
	decalage = 0
	for i in listeIdAttributs:
		if i<idAttribut:
			decalage+=dicoDesAttributs[i]['dimensions']
	return decalage


def importerDonneesBrutes(
		sourcerJC = False,
		dirJC = DIR_JC,
		sourcerJSY = False,
		dirJSY = DIR_JSY,
	):
	r"""
	\brief
	Lit fichiers de données brutes et récolte valeurs dans pandas DataFrames.

	\param[in] sourcerJC
	Vrai si on doit utiliser les fichiers de données
	brutes de Jonathan Cauchon, faux sinon.

	\param[in] dirJC
	Répertoire contenant données brutes de Jonathan Cauchon.

	\param[in] sourcerJSY
	Vrai si on doit utiliser les fichiers de données
	brutes de Jonathan St-Yves, faux sinon.

	\param[in] dirJSY
	Répertoire contenant données brutes de Jonathan St-Yves.

	\return
	Listes de couples (idAttributs,DataFrame), un couple pour chaque
	fichier lu; les itérables idAttributs sont ordonnés selon les
	attributs présents dans la DataFrame.
	"""
	print("IMPORTATION DES DONNÉES BRUTES")

	donnees=[]

	INTER = ' '
	EN_TETE = '{:20s}'.format("répertoire")+INTER+\
		'{:30s}'.format("fichier")+INTER+\
		'{:16s}'.format("(r,c)")
	print()
	print(EN_TETE)
	print('-'*65)

	# FIXME, TODO: ne pas répéter deux fois; modulariser selon source (JSY|JC)
	if sourcerJC:

# ???, FIXME: les trois (3) premières lignes de données du fichier de JC
#	    disponible sur le site github manquent une colonne de donnée;
#       il n'y en a que 4208 alors qu'il devrait y en avoir 4209


# ???, FIXME: l'apodisation telle que générée par le script de JC
#		commence par une valeur nulle à la première position, ce qui
#		force l'emploi de domaine de validation R+ au lieu de R+*;
#		Est-ce normal? En comparison, les données d'apodisation générées
#		par JSY sont exclusivement limitées au domaine R+*.


		assert(os.path.exists(os.path.normpath(dirJC))) # check

		fichiersJC = [f for f in os.listdir(os.path.normpath(dirJC))
						if f.endswith(EXT_JC)]

		assert(len(fichiersJC)>0) # check


		for nomFichier in fichiersJC:
			df0=pandas.read_csv(
					os.path.normpath(os.path.join(
						dirJC,nomFichier)),
					sep='\s+',
					header=None,
					skiprows=1,
			)
			donnees.append((ATTRIBUTS_JC,df0))
			EN_TETE = '{:20s}'.format(dirJC)+INTER+\
				'{:30s}'.format(nomFichier)+INTER+\
				'{:16s}'.format(str(df0.shape))
			print(EN_TETE)


	if sourcerJSY:
		assert(os.path.exists(os.path.normpath(dirJSY))) # check

		fichiersJSY = [f for f in os.listdir(os.path.normpath(dirJSY))
						if f.endswith(EXT_JSY)]

		assert(len(fichiersJSY)>0) # check

		for nomFichier in fichiersJSY:
			df0=pandas.read_csv(
					os.path.normpath(os.path.join(
						dirJSY,nomFichier)),
					sep='\s+',
					header=None,
					skiprows=1,
			)
			donnees.append((ATTRIBUTS_JSY,df0))
			EN_TETE = '{:20s}'.format(dirJSY)+INTER+\
				'{:30s}'.format(nomFichier)+INTER+\
				'{:16s}'.format(str(df0.shape))
			print(EN_TETE)

	print()
	return donnees


def refondreDonnees(donneesBrutes):
	r"""
	\brief
	Transforme les DataFrames de données brutes en un
	seul dictionnaire de données structurées.

	\param[in] donneesBrutes
	Liste de couples (idAttributs,DataFrame). Il y a autant
	de couples que de fichiers de données brutes lus.

	\return
	Dictionnaire de données structurées; les clés principales sont
	des entiers naturels successifs (un par donnée distincte); à chaque
	clé principale est attaché un sous-dictionnaire de valeurs; les
	clés de ce sous-dictionnaires correspondent aux identifiants des
	différents attributs provenant des données brutes et présents
	pour cette donnée.

	\note
	Selon la quantité de dimensions des attributs, les valeurs contenues
	dans les sous-dictionnaires sont soit des nombres (si la quantité de
	dimensions est 1), soit des 	NumPy array de nombres (si la quantité
	de dimensions est plus de 1).
	"""
	print("REFONTE DES DONNÉES DANS STRUCTURE UNIFIÉE")
	print()

	# initialiser dictionnaire unifié des valeurs
	dicoDesValeurs = {}

	# pour chaque fichier de données chargé dans une DataFrame
	nbDataFrames = len(donneesBrutes)
	# pour formatter l'écriture du compteur de fichier
	nbDigits = len(str(nbDataFrames))
	compteurFichiers=0
	idDonnee=0
	for attributs,df in donneesBrutes:
		compteurFichiers+=1
		# pour chaque rangée de la DataFrame
		print("---> depuis fichier {0:{n}d}/{1}".format(
				compteurFichiers,nbDataFrames,n=nbDigits))
		for i in numpy.arange(len(df)):
			# récupérer cette rangée (ligne) de données brutes
			ligne = df.iloc[i]
			# créer un dictionnaire ayant les mêmes clés principales
			# que dicoDesAttributs mais un seul champ numérique par attribut
			dicoValeursPourUneDonnee = {}
			# le compléter avec les valeurs associées à chaque attribut
			for k in attributs:
				# retenir le domaine de la valeur de cet attribut
				domaine = dicoDesAttributs[k]['domaine']
				# calculer le décalage en colonnes des valeurs
				decalage = decalageEnColonnes(k,attributs)
				# retenir le nombre de valeurs à charger
				dimensions = dicoDesAttributs[k]['dimensions']
				# si plus d'une dimension, employer un NumPy array
				# sinon, enregistrer la valeur directement
				if dimensions>1:
					# préparer un NumPy array
					plusieursValeurs = numpy.array([])
					# configurer son type numérique
					if 'REEL' in domaine:
						plusieursValeurs.astype(float)
					elif 'NAT' in domaine:
						plusieursValeurs.astype(int)
					else:
						raise

					for j in numpy.arange(dimensions):
						# extraire la valeur
						valeur = ligne[j+decalage]
						# valider le domaine de la valeur
						try:
							assert(domaineValide(valeur,domaine))
						except:
							print(
								"!!! domaineValide_plusieursValeurs"
								"(i,k,j,decalage) = (%d,%d,%d,%d)"
								%(i,k,j,decalage))
						# ajouter cette valeur au NumPy array
						plusieursValeurs = numpy.append(
								plusieursValeurs,valeur)

					# ajouter une valeur correspondant à cet attribut
					dicoValeursPourUneDonnee.update({k:plusieursValeurs})

				else:
					assert(dimensions==1) # check
					# extraire la valeur
					uneValeur = ligne[decalage]
					# valider le domaine de la valeur
					try:
						assert(domaineValide(uneValeur,domaine))
					except:
						print(
							"!!! domaineValide_uneValeur"
							"(i,k,decalage) = (%d,%d,%d)"
							%(i,k,decalage))
					# forcer l'emploi d'un entier si approprié
					if domaine == 'NAT' or domaine == 'NATPOS':
						uneValeur = int(uneValeur)
					# ajouter une valeur correspondant à cet attribut
					dicoValeursPourUneDonnee.update({k:uneValeur})

			# ajouter le dictionnaire rempli pour cette donnée
			# au dictionnaire unifié des valeurs
			dicoDesValeurs.update({idDonnee:dicoValeursPourUneDonnee})
			idDonnee+=1

	print()
	# retourner dictionnaire unifié des valeurs
	return dicoDesValeurs


def enregistrementPickle(dicoDesValeurs,pfxPickle=""):
	r"""
	\brief
	Enregistre un dictionnaire de données structurées dans un
	fichier binaire Pickle.

	\details
	Le dictionnaire enregistré consiste en deux dictionnaires distincts
	mais inter-reliés: le premier contient la description des attributs
	et le second contient des valeurs associées à ces attributs; ces
	valeurs sont groupées dans leur propre dictionnaire, un par
	donnée enregistrée.

		d = {d1:{},d2:{}}

			P = nbAttributs - 1
			Q = nbTotalDonnees - 1

		d1 = {0:da0,1:da1,...,P:daP} <--- cf. dicoDesAttributs
		d2 = {0:dv0,1:dv1,...,Q:dvQ} <--- cf. dicoDesValeurs

			da<p>.keys() = ['symbole','unites','description',
							 'domaine','dimensions']

			dv<q>.keys() = [0,1,...,P]

	\param[in] dicoDesValeurs
	Dictionnaire des valeurs à enregistrer.

	\param[in] pfxPickle
	Préfixe à donner au nom de fichier binaire Pickle.

	\post
	Les dictionnaires d'attributs et de valeurs ont été enregistrés
	dans un fichier binaire en format Pickle. Leur re-chargement a été
	testé et procure les mêmes objets.

	\note
	Un test d'intégrité est effectué afin de vérifier que les valeurs
	peuvent être rechargées par une fonction dédiée et qu'elles sont
	alors les mêmes que celles existant avant l'enregistrement.
	"""
	print("ENREGISTREMENT EN FORMAT BINAIRE PICKLE")

	dicoPourEnregistrer = {
			'lesAttributs':dicoDesAttributs,
			'lesValeurs':dicoDesValeurs,
	}

	if not os.path.exists(os.path.normpath(DIR_STRUCT)):
		os.makedirs(os.path.normpath(DIR_STRUCT))

	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	with open(nomFichier, 'wb') as fichierPkl:
	    pickle.dump(
				dicoPourEnregistrer,
				fichierPkl,
				protocol=pickle.HIGHEST_PROTOCOL)

	# test de rechargement
	dicoReCharge = chargementPickle(pfxPickle,silencieux=True)

	# test d'intégrité
	try:
		assert(dicoReCharge['lesAttributs'] ==
			dicoPourEnregistrer['lesAttributs'])

		for k1 in dicoReCharge['lesValeurs'].keys():
			for k2 in dicoReCharge['lesValeurs'][k1].keys():
				if dicoReCharge['lesAttributs'][k2]['dimensions']>1:
					assert(numpy.all(dicoReCharge['lesValeurs'][k1][k2]==
						 dicoPourEnregistrer['lesValeurs'][k1][k2]))
				else:
					assert(dicoReCharge['lesValeurs'][k1][k2]==
						  dicoPourEnregistrer['lesValeurs'][k1][k2])
	except:
		print("!!! recharge Pickle non-consitante")


def chargementPickle(pfxPickle="",silencieux=False):
	r"""
	\brief
	Fonction dédiée pour la recharge d'un dictionnaire de données
	structurées depuis fichier binaire Pickle.

	\param[in] pfxPickle
	Préfixe à donner au nom de fichier binaire Pickle.

	\param[in] silencieux
	Si vrai, alors ne pas imprimer de message; si faux, alors
	imprimer message qui avertit l'usager que la recharge depuis
	fichier binaire Pickle est effectuée.

	\return
	Dictionnaire de données structurées dont la structure est
	décrite dans les détails de la documentation de la
	fonction enregistrementPickle().
	"""
	if not silencieux:
		print("CHARGEMENT DEPUIS FORMAT BINAIRE PICKLE")
	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	with open(nomFichier, 'rb') as fichierPkl:
	    dicoDonneesRechargees = pickle.load(fichierPkl)

	return dicoDonneesRechargees


def retrouverIdDunSymbole(symbole):
	r"""
	\brief
	Retrouve l'identifiant numérique de l'attribut relié à un symbole.

	\param[in] symbole
	Chaîne de caractères qui identifie un attribut.

	\return
	Identifiant numérique de l'attribut ayant le symbole demandé;
	None s'il n'y a pas eu de correspondance.
	"""
	nbAttributs = len(dicoDesAttributs.keys())
	i=0
	while dicoDesAttributs[i]['symbole']!=symbole and i<nbAttributs:
		i+=1
	if i<nbAttributs:
		return i
	return None


def afficherProgression(
    iteration,
    total,
    prefixe='--->',
    suffixe='Complété',
    decimales=1,
    longueur=40,
    symbole='█'):
	r"""
	\brief
	Indicateur de progression pour affichage console

	\param[in] iteration
	Itération actuelle de la progression affichée [0..total]

    \param[in] total
	Nombre total d'itérations sur lequel le calcul de progression est basée

	\param[in] prefixe
	Texte à écrire à gauche de la barre de progression

	\param[in] suffixe
	Texte à ajouter à droite de la barre de progression

	\param[in] decimales
	Nombre de décimales à afficher après le point

	\param[in] longueur
	Longueur de la barre de progression

	\param[in] symbole
	Caractère à utiliser pour remplir la barre de progression

	\warning
	Dans les consoles ipython (et autres?) sous windows, la ligne de
	progression se dédouble...
	"""
	# FIXME: faire en sorte que la ligne de se dédouble pas dans windows
	pourcentage=("{0:."+str(decimales)+"f}").format(100*iteration/total)
	rempli=int(longueur*iteration/total)
	barre=symbole*rempli+'-'*(longueur-rempli)
	print('\r%s |%s| %s%% %s' %(prefixe,barre,pourcentage,suffixe),end="")
	if iteration==total:
		print()


def creerEchantillonnage(
		dicoValeurs,
		idAttribX,
		idAttriby,
		amplitude=False,
		phase=False):
	r"""
	\brief
	Crée un échantillonnage de données et d'étiquettes.

	\param[in] dicoValeurs
	Dictionnaire contenant uniquement des *VALEURS* pour
	chaque attribut et ce pour toutes les données disponibles.

	\param[in] idAttribX
	Liste des identifiants d'attributs choisis pour constituer
	les données d'entrée.

	\param[in] idAttriby
	Liste des identifiants d'attributs choisis pour constituer
	les étiquettes des données d'entrée.

	\param[in] amplitude
	Drapeau indiquant si on doit combiner les deux seuls attributs
	contenus dans idAttribX pour construire une valeur réelle représantant
	l'amplitude du nombre complexe formé par les deux données fournies.

	\param[in] phase
	Drapeau indiquant si on doit combiner les deux seuls attributs
	contenus dans idAttribX pour construire une valeur réelle représentant
	la phase du nombre complexe formé par les deux données fournies.

	\return
	Tuple (X,y) de deux NumPy array contenant en X les valeurs de
	données d'entrée et en y les valeurs des étiquettes reliées
	à ces données d'entrée. La forme de X et de y est respectivement
	(nbTotalDonnees,nbValeurs{X|y}) où nbValeursX est le nombre de
	valeurs numériques distinctes contenues 	dans chaque donnée X et
	nbValeursy est le nombre de valeurs 	numériques distinctes
	contenues dans chaque étiquette y.
	Si n'y a aucune donnée de compatible avec les attributs
	demandés, alors retourne None.

	\warning
	Les données disponibles doivent être compatibles avec
	les attributs demandés.
	"""
	print("CRÉATION DE L'ÉCHANTILLONNAGE")
	print()

	# afin d'avoir des itérables de type "liste"
	listeIdX = list(idAttribX)
	listeIdy = list(idAttriby)


	# il faudra seulement sélectionner parmi les données disponibles
	# celles qui sont compatibles avec les attributs demandés!

	# les données X et les étiquettes y doivent être des ensembles disjoints
	try:
		assert(idAttribX.isdisjoint(idAttriby))
	except:
		print("!!! attributs non disjoints")

	# construire l'ensemble des attributs demandés
	attributsDemandes = idAttribX.union(idAttriby)

	X = numpy.array([])
	y = numpy.array([])

	# ??? on veut calculer soit l'amplitude, soit la phase, mais
	#     pas les deux; cette hypothèse pourrait sûrement être relaxée,
	#     mais limitons-nous à cette restriction pour l'instant...
	try:
		assert((amplitude and phase) == False)
	except:
		print("!!! amplitude OU phase")

	complexe = amplitude or phase
	if complexe:
		# il ne doit y avoir que deux attributs dans les données
		try:
			assert(len(idAttribX)==2)
		except:
			print("!!! deux valeurs requises (en X?) pour nombre complexe")
		# l'attribut réel doit être le premier
		# !!! adHoc: documenter
		idRe = listeIdX[0]
		# l'attribut imaginaire doit être le second
		# !!! adHoc: documenter
		idIm = listeIdX[-1]

	nbDeDonneesCompatibles = 0
	nbDicoValeurs = len(dicoValeurs)
	afficherProgression(0,nbDicoValeurs)

	# pour chaque donnée du dictionnaire de valeurs fourni
	for i in numpy.arange(nbDicoValeurs):

		# si les attributs de cette donnée sont
		# compatibles avec ceux demandés
		if set(dicoValeurs[i].keys()).\
			intersection(attributsDemandes)\
				==attributsDemandes:

			dX = numpy.array([])

			if complexe:
				# calculer soit l'amplitude, soit la phase du nombre complexe
				# et ajouter les valeurs calculées à dX
				re = dicoValeurs[i][idRe]
				im = dicoValeurs[i][idIm]
				if amplitude:
					dX = numpy.sqrt(re**2+im**2)
				else:
					dX = numpy.arctan2(im,re)
			else:
				# pour chaque attribut de donnée
				for idX in listeIdX:
					# ajouter les valeurs à dX
					dX = numpy.append(dX,dicoValeurs[i][idX])

			# ajouter dX à X
			X = numpy.append(X,dX)

			dy = numpy.array([])

			# pour chaque attribut d'étiquette
			for idy in listeIdy:
				# ajouter les valeurs à dy
				dy = numpy.append(dy,dicoValeurs[i][idy])

			# ajouter dy à y
			y = numpy.append(y,dy)

			nbDeDonneesCompatibles+=1

		afficherProgression(1+i,nbDicoValeurs)

	print()


	if nbDeDonneesCompatibles == 0:
		print("!!! Attributs demandés incompatbles "
		"avec données refondues disponibles.")
		raise

	nbValeursX = int(len(X) / nbDeDonneesCompatibles)
	nbValeursy = int(len(y) / nbDeDonneesCompatibles)

	X=X.reshape((nbDeDonneesCompatibles,nbValeursX))
	y=y.reshape((nbDeDonneesCompatibles,nbValeursy))

	# vérifications de la consistance de la forme des NumPy array

	try:
		assert(X.ndim == 2)
		assert(y.ndim == 2)
	except:
		print("!!! inconsistance {X|y}.ndim")

	try:
		assert(X[0].ndim == 1)
		assert(y[0].ndim == 1)
	except:
		print("!!! inconsistance {X|y}[0].ndim")

	return X,y


def enregistrementPyTorch(X,y,pfxPyTorch):
	r"""
	\brief
	Enregistre les données et leurs étiquettes en tenseurs PyTorch.

	\param[in] X
	NumPy array des données.

	\param[in] y
	NumPy array des étiquettes.

	\param[in] pfxPyTorch
	Préfixe de nom à donner aux fichiers PyTorch.

	\post
	Les données X et les étiquettes y ont été convertis en
	tenseurs PyTorch et enregistrées dans un format directement
	utilisable par PyTorch. Leur re-chargement a été
	testé et procure les mêmes objets.
	"""
	# !!! il n'est pas possible de traiter directement des
	#     nombres complexes avec les modules standard de PyTorch;
	#     voir la note en en-tête du script
	try:
		assert(type(X)!=complex)
		assert(type(y)!=complex)
	except:
		print("!!! variables complexes non-supportées par PyTorch")

	print("ENREGISTREMENT EN FORMAT PYTORCH")

	if not os.path.exists(os.path.normpath(DIR_STRUCT)):
		os.makedirs(os.path.normpath(DIR_STRUCT))

	nomFichierX = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_X))

	nomFichiery = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_y))

	tX = torch.from_numpy(X)
	ty = torch.from_numpy(y)

	torch.save(tX,nomFichierX)
	torch.save(ty,nomFichiery)

	# checks: vérification que ce qui est rechargé est la
	# même chose que ce qui était présent avant l'enregistrement

	rtX = torch.load(nomFichierX)
	rty = torch.load(nomFichiery)

	try:
		assert(torch.all(torch.eq(rtX,tX)))
		assert(torch.all(torch.eq(rty,ty)))
	except:
		print("!!! recharge PyTorch non-consistante")


class Echantillonnage(torch.utils.data.Dataset):
	r"""
	\brief
	Classe qui instancie des objets dérivés de torch.utils.data.Dataset

	\details
	Les objets de la classe Echantillonnage sont spécialisés
	pour lire les fichiers de données créés pour PyTorch par
	le présent script de traitement.
	Ces fichiers contiennent la transformation des valeurs brutes
	provenant des simulations de réseaux contra-directionnels par
	méthode de matrice de transfert.
	Les valeurs ont été restructurées et manipulées pour être
	séparées en données X et en 	étiquettes y puis entreposées
	sous forme de tenseurs.
	L'ensemble de ces tenseurs forme l'échantillonnage.

	\note
	Entre le {map|iterable}-style datasets, on implémente ici
	le map-style dataset dans lequel on doit redéfinir les
	méthodes __getitem__() et __len__()
	"""
	def __init__(self,pfxPyTorch=""):
		r"""
		\brief
		Initialisateur des objets instanciés

		\param[in] pfxPyTorch
		Préfixe à appliquer aux noms de fichiers PyTorch

		\details
		Appelle premièrement le contructeur de la classe de base
		et ensuite définit en quoi consiste le constructeur
		spécialisé de la présente classe dérivée.
		"""
		super().__init__()

		nomFichierX = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_X))

		nomFichiery = os.path.normpath(os.path.join(
			DIR_STRUCT,
			pfxPyTorch+NOM_PYTORCH_y))

		# charger les données dans une variable membre
		self.X = torch.load(nomFichierX)

		# charger les étiquettes dans une variable membre
		self.y = torch.load(nomFichiery)

		# véfification basique de la consistance entre X et y
		assert(len(self.X)==len(self.y))

		# enregistrer la cardinalité de l'échantillonnage
		self.nbTotalDonnes = self.__len__()


	def __getitem__(self, i):
		r"""
		\brief
		Retourne un des échantillons de l'échantillonnage.

		\param[in] i
		Indice de l'échantillon demandé. Ce doit être un entier
		de zéro à la cardinalité de l'échantillonnage.

		\warning
		Si un remodelage des données est nécessaire avant
		son emploi dans un réseau de neurones, cela devrait
		probablement être effectué dans cette méthode.
		Si cette méthode est implémentée hors de la portée
		du présent script, alors les paramètres communs et
		indispensables au remodelage tels NB_LAMBDAS et NB_BLOCS
		devraient être accessibles depuis un module de
		paramètres communs.
		"""
		assert(domaineValide(i,'NAT'))
		assert(i<self.__len__())
		return self.X[i]


	def __len__(self):
		r"""
		\brief
		Retourne la cardinalité de l'échantillonnage.
		"""
		return len(self.X)



def illustrerEchantillonnage(chi,idAttribX,idAttriby):
	r"""
	\brief
	Trace le graphique de quatre données d'entrée tirées au hasard.


	\param[in] chi
	Instance de la classe Echantillonnage.


	\param[in] idAttribX
	Identifiants des attributs de données.

	\param[in] idAttriby
	Identifiants des attributs d'étiquettes.


	\post
	Une figures avec 4 sous-graphiques est tracée sur laquelle
	on a les indices i des données X tracées ainsi que la valeur de
	leurs étiquettes y.
	"""

	print("ILLUSTRATION DE L'ÉCHANTILLONNAGE")

	# l'affichage 2x2 d'attributs multi-dimensionnels
	# suppose au moins 4 données disponibles
	assert(len(chi)>=4)

	# déterminer parmi l'union des idAttrib quels sont ceux de dimension>1
	idPourGraphes = set()
	for a in idAttribX.union(idAttriby):
		if dicoDesAttributs[a]['dimensions']>1:
			idPourGraphes.add(a)

	# pour chaque attribut multi-dimensionnel
	for gId in idPourGraphes:

		# retrouver le symbole texte associé à cet attribut
		symbole = dicoDesAttributs[gId]['symbole']

		# retrouver le nombre de dimensions (éléments) de cet attribut
		nbElements = dicoDesAttributs[gId]['dimensions']

		# retrouver l'ensemble des valeurs pour cet attribut dans chi

		# si cet attribut est employé comme donnée X
		if gId in idAttribX:
			usage = 'X'
			beg = decalageEnColonnes(gId,list(idAttribX))
			end = beg+nbElements
			lesVals = chi.X[:,beg:end].numpy()

		# si cet attribut est employé comme étiquette y
		if gId in idAttriby:
			usage = 'y'
			beg = decalageEnColonnes(gId,list(idAttriby))
			end = beg+nbElements
			lesVals = chi.y[:,beg:end].numpy()

		# dessiner une figure avec 4 sous-graphes
		fig, subfigs = matplotlib.pyplot.subplots(
				2, # nombre de rangées
				2, # nombre de colonnes
				tight_layout=False)

		# choisir 4 données distinctes en ordre d'indice croissant
		idx = numpy.random.randint(0,len(chi),size=1)
		while len(idx)<4:
		        i = numpy.random.randint(0,len(chi))
		        if not i in idx:
		                idx = numpy.append(idx,i)
		idx.sort()

		# remplir et annoter chaque sous-graphe
		for i,subfig in zip(idx,subfigs.reshape(-1)):

			ANNOT = ""

			# retrouver les valeurs d'attributs de dimension
			# unitaire reliés à cette donnée

			for gId in idAttribX:
				if gId not in idPourGraphes:
					if len(ANNOT)>0:
						ANNOT+="\n"
					ANNOT+="X::"+dicoDesAttributs[gId]['symbole']+" = "
					dmn = dicoDesAttributs[gId]['domaine']
					uni = dicoDesAttributs[gId]['unites']
					poz = decalageEnColonnes(gId,list(idAttribX))
					vals = chi.X[:,poz].numpy()
					v = vals[i]
					if 'NAT' in dmn:
						# alors le domaine est "NAT[*]"
						dformat = 'd'
						v=int(v)
						nbPlaces=len(str(v))
					else:
						# alors le domaine est "REEL[+][*]"
						dformat = '.2f'
						nbPlaces=3+str(v).index('.')
					ANNOT+= '{:<{plc}{dom}}'.\
						format(v,plc=nbPlaces,dom=dformat)
					ANNOT+=" "+uni

			for gId in idAttriby:
				if gId not in idPourGraphes:
					if len(ANNOT)>0:
						ANNOT+="\n"
					ANNOT+="y::"+dicoDesAttributs[gId]['symbole']+" = "
					dmn = dicoDesAttributs[gId]['domaine']
					uni = dicoDesAttributs[gId]['unites']
					poz = decalageEnColonnes(gId,list(idAttriby))
					vals = chi.y[:,poz].numpy()
					v = vals[i]
					if 'NAT' in dmn:
						# alors le domaine est "NAT[*]"
						dformat = 'd'
						v=int(v)
						nbPlaces=len(str(v))
					else:
						# alors le domaine est "REEL[+][*]"
						dformat = '.2f'
						nbPlaces=3+str(v).index('.')
					ANNOT+= '{:<{plc}{dom}}'.\
						format(v,plc=nbPlaces,dom=dformat)
					ANNOT+=" "+uni

			subfig.scatter(numpy.arange(len(lesVals[i])),lesVals[i])
			subfig.set_title("(i)=(%g)"%(i))
			subfig.set_axis_off()

			subfig.annotate(
				ANNOT,
				xy=(0.05,0.95),
				xycoords='axes fraction',
				horizontalalignment='left',
				verticalalignment='bottom',
				bbox = dict(boxstyle="round", fc="0.65"))

		fig.suptitle("Affichage de quatre données aléatoires\n"
					 "parmi l'échantillonnage de %d données\n"
					 "pour l'attribut %s::%s"
					 %(chi.nbTotalDonnes,usage,symbole))
		fig.show()


def statsEchantillonnage(chi,idAttribX,idAttriby):
	r"""
	\brief
	Trace le graphique des statistiques de certains attributs de
	l'échantillonnage.


	\param[in] chi
	Instance de la classe Echantillonnage.


	\param[in] idAttribX
	Identifiants des attributs de données.

	\param[in] idAttriby
	Identifiants des attributs d'étiquettes.


	\post
	Une figures avec autant de sous-graphiques que d'attributs de
	dimension unitaire est tracée sur laquelle on a, pour chaque
	attribut tracé, le résumé des valeurs statistiques min, max, moy et std.
	"""

	print("STATISTIQUES DE L'ÉCHANTILLONNAGE")

	print()
	print("---> Nombre total de données = %d"%(chi.nbTotalDonnes))
	print()

	# commencer par faire un tableau texte à la console


	INTER = ' '*2
	EN_TETE='{:>10s}'.format("attribut")+INTER+\
			'{:>5s}'.format("usage")+INTER+\
			'{:>10s}'.format("min")+INTER+\
			'{:>10s}'.format("max")+INTER+\
			'{:>10s}'.format("moy")+INTER+\
			'{:>10s}'.format("std")+'\n'
	LIGNE = '-'*len(EN_TETE[:-1])
	EN_TETE+=LIGNE
	print(EN_TETE)

	# déterminer parmi l'union des idAttrib quels sont ceux de dimension=1
	idPourStats = set()
	for a in idAttribX.union(idAttriby):
		if dicoDesAttributs[a]['dimensions']==1:
			idPourStats.add(a)

	nbGraphes = len(idPourStats)
	# FIXME: problème avec subplots lorsque nbGraphes = 1
	assert(nbGraphes>1)


	fig, subfigs = matplotlib.pyplot.subplots(
			1, # nombre de rangées
			nbGraphes, # nombre de colonnes
			tight_layout=False)

	for sId in idPourStats:
		# retrouver symbole attaché à cet attribut
		symbole = dicoDesAttributs[sId]['symbole']
		domaine = dicoDesAttributs[sId]['domaine']
		unites = dicoDesAttributs[sId]['unites']
		# extraire ensemble des valeurs de cet attribut
		if sId in idAttribX:
			pos = list(idAttribX).index(sId)
			lesVals = chi.X[:,pos].numpy()
			usage = 'X'
		if sId in idAttriby:
			pos = list(idAttriby).index(sId)
			lesVals = chi.y[:,pos].numpy()
			usage = 'y'
		# calculer des statistiques
		valmin = lesVals.min()
		valmax = lesVals.max()
		moyenne = lesVals.mean()
		nbPlacesMoyenne=3+str(moyenne).index('.')
		stddev = lesVals.std()
		nbPlacesStd=3+str(stddev).index('.')
		if 'NAT' in domaine:
			# alors le domaine est "NAT[*]"
			dformat = 'd'
			valmin = int(valmin)
			nbPlacesValmin=len(str(valmin))
			valmax = int(valmax)
			nbPlacesValmax=len(str(valmax))

		else:
			# alors le domaine est "REEL[+][*]"
			dformat = '.2f'
			nbPlacesValmin=3+str(valmin).index('.')
			nbPlacesValmax=3+str(valmax).index('.')


		# chaîne de caractères pour l'annotation de cette sous-figure
		ANNOT="min = "+'{:<{plc}{dom}}'.\
					format(valmin,plc=nbPlacesValmin,dom=dformat)+'\n'+\
				"max = "+'{:<{plc}{dom}}'.\
					format(valmax,plc=nbPlacesValmax,dom=dformat)+'\n'+\
				"moy = "+'{:<{plc}.2f}'.\
					format(moyenne,plc=nbPlacesMoyenne)+'\n'+\
				"std = "+'{:<{plc}.2f}'.\
					format(stddev,plc=nbPlacesStd)

		subfigs[pos].hist(lesVals)
		subfigs[pos].set_title(usage+"::"+symbole)
		subfigs[pos].set_xlabel(unites)
		subfigs[pos].annotate(
				ANNOT,
				xy=(0.15,0.05),
				xycoords='axes fraction',
				horizontalalignment='left',
				verticalalignment='bottom',
				bbox = dict(boxstyle="round", fc="0.65"))

		# chaîne de caractères pour le tableau statistique console
		STATS='{:>10s}'.format(symbole)+INTER+\
				'{:>5s}'.format(usage)+INTER+\
				'{:>10{dom}}'.format(valmin,dom=dformat)+INTER+\
				'{:>10{dom}}'.format(valmax,dom=dformat)+INTER+\
				'{:>10.2f}'.format(moyenne)+INTER+\
				'{:>10.2f}'.format(stddev)


		print(STATS)

	print()

	fig.suptitle("Répartition des valeurs d'attributs de dimension\n"
			  "unitaire dans l'échantillonnage de %d données"
			  %(chi.nbTotalDonnes))
	fig.show()


def procederAuTraitement(
		forcerImportation=False,
		sourcerJC=False,
		repertoireJC=DIR_JC,
		sourcerJSY=False,
		repertoireJSY=DIR_JSY,
		idAttribX = ATTRIBUTS_X,
		idAttriby = ATTRIBUTS_y,
		pfxPickle = "",
		pfxPyTorch = "",):
	r"""
	\brief
	Simule la chaîne de processus de traitement en un bloc

	\param[in] forcerImportation (bool)
	Force l'importation de données brutes pour créer un Pickle.

	\param[in] sourcerJC (bool)
	Utiliser les données brutes de Jonathan Cauchon

	\param[in] repertoireJC
	Répertoire où se trouvent les données brutes de Jonathan Cauchon

	\param[in] sourcerJSY
	Utiliser les données brutes de Jonathan St-Yves

	\param[in] repertoireJSY
	Répertoire où se trouvent les données brutes de Jonathan St-Yves

	\param[in] idAttribX
	Identifiants d'attributs constituants les données X

	\param[in] idAttriby
	Identifiants d'attributs constituants les étiquettes y

	\param[in] pfxPickle
	Préfixe du nom de fichier Pickle

	\param[in] pfxPyTorch
	Préfixe des noms de fichiers PyTorch

	\details
	La chaîne de processus qui permet de préparer les
	données brutes pour chargement PyTorch consiste à
		(1) importer les données brutes,
		(2) refondre les données dans une structure unifiée,
		(3) enregistrer la refonte en Pickle,
		(4) choisir des attributs de données et des attributs d'étiquettes
		(5) créer un échantillonnage en fonction des choix d'attributs,
		(6) enregistrer l'échantillonnage en tenseurs PyTorch,
		(7) instancier un objet dérivé de torch.utils.data.Dataset
		(8) illustrer des exemples échantillons parmi l'échantillonnage
		(9) calculer des statistiques sur l'échantillonnage

	\note
	La pré-existence d'un fichier Pickle correspondant au nom
	demandé ainsi que la combinaison des arguments optionnels du
	script permettent de sauter les étapes d'importation, de
	refonte et d'enregistrement Pickle (1,2,3) qui peuvent
	être longues avec un volume important de fichiers de
	données brutes.

	\note
	Le symbole employé dans le cours pour désigner l'échantillonnage
	global est $\mathcal{X}$, ce qui ressemble beaucoup à $\chi$.
	C'est pourquoi j'utilise "chi" comme nom de variable pour
	instancier un objet Echantillonnage dérivé
	de torch.utils.data.Dataset.


	\todo
	# ???, FIXME, TODO: implémenter sélection amplitude de E_drop

	\todo
	# ???, FIXME, TODO: implémenter sélectionner phase de E_drop
	"""


	nomFichier = os.path.normpath(os.path.join(
			DIR_STRUCT,pfxPickle+NOM_PICKLE))

	if os.path.exists(nomFichier) and not forcerImportation:
		dicoDonneesRechargees = chargementPickle(pfxPickle)
		dicoDesValeurs = dicoDonneesRechargees['lesValeurs']
	else:
		if not (sourcerJC or sourcerJSY):

			print("!!! Fichier Pickle absent et aucune source demandée ...")
			return

		else:
			# (1) importer les données brutes,
			donneesBrutes = importerDonneesBrutes(
					sourcerJC,repertoireJC,
					sourcerJSY,repertoireJSY)
			# (2) refondre les données dans une structure unifiée,
			dicoDesValeurs = refondreDonnees(donneesBrutes)
			# (3) enregistrer la refonte en Pickle,
			enregistrementPickle(dicoDesValeurs,pfxPickle)



	# (4) choisir des attributs de données et des attributs d'étiquettes
	#
	#     ---> selon les arguments -X et -y ou ceux par défaut
	#

	# (5) créer un échantillonnage en fonction des choix d'attributs
	try:
		X,y = creerEchantillonnage(
				dicoDesValeurs,
				idAttribX,
				idAttriby,
				amplitude=False, # effet à implémenter !
				phase=False) # effet à implémenter !
	except:

		print("!!! Impossible de créer échantillonnage ...")
		return

	# (6) enregistrer l'échantillonnage en tenseurs PyTorch
	enregistrementPyTorch(X,y,pfxPyTorch)

	# (7) instancier un objet dérivé de torch.utils.data.Dataset
	chi = Echantillonnage(pfxPyTorch)

	print()
	print("---> Nombre total de données = %d"%(chi.nbTotalDonnes))
	print()

	# NOTE: illustration de l'instanciation d'un objet
	#       dérivé de torch.utils.data.Dataset servant à
	#       l'entraînement par lots; pas utilisé ici...
	chargeur = torch.utils.data.DataLoader(chi, batch_size=16)


	# (8) illustrer des exemples échantillons parmi l'échantillonnage
	illustrerEchantillonnage(chi,idAttribX,idAttriby)

	# (9) calculer des statistiques sur l'échantillonnage
	statsEchantillonnage(chi,idAttribX,idAttriby)



def gererArguments():
	r"""
	\brief
	Traite les arguments fournis au script sur la ligne de commande
	"""
	INTER = ' '*2
	LISTE_ATTRIBUTS='\n\n'+'{:>3s}'.format("id")+INTER+\
			'{:16s}'.format("attribut")+INTER+\
			'{:16s}'.format("usage possible")+'\n'
	LIGNE = '-'*len(LISTE_ATTRIBUTS[:-1])
	LIGNEn = LIGNE+'\n'
	LISTE_ATTRIBUTS+=LIGNEn
	XSet = set()
	ySet = set()
	for k in dicoDesAttributs.keys():
		XSet.add(k)
		ySet.add(k)
		LISTE_ATTRIBUTS+='{:3d}'.format(k)+INTER+\
				'{:16s}'.format(dicoDesAttributs[k]['symbole'])+INTER+\
				'{:16s}'.format(dicoDesAttributs[k]['usage'])+'\n'

	analyseur = argparse.ArgumentParser(
			formatter_class=argparse.RawDescriptionHelpFormatter,
			description=textwrap.indent(
"""
Introduction à l'apprentissage machine
ULaval, GIF-7005-85309, A2019, équipe 13
%s
Script de traitement des données brutes
et de génération d'échantillonnages (X,y)
compatibles avec PyTorch afin de procéder
à l'entraînement de réseaux de neurones.


 Valeur paramètres par défaut du script
%s
         DIR_JC = %s
         EXT_JC = %s
        DIR_JSY = %s
        EXT_JSY = %s
     DIR_STRUCT = %s
     NOM_PICKLE = %s
    ATTRIBUTS_X = %s
    ATTRIBUTS_y = %s
  NOM_PYTORCH_X = %s
  NOM_PYTORCH_y = %s

%s"""%(
			LIGNE,
			LIGNE,
			DIR_JC,
			EXT_JC,
			DIR_JSY,
			EXT_JSY,
			DIR_STRUCT,
			NOM_PICKLE,
			ATTRIBUTS_X,
			ATTRIBUTS_y,
			NOM_PYTORCH_X,
			NOM_PYTORCH_y,
			LISTE_ATTRIBUTS),"     "))


	groupeJC = analyseur.add_argument_group(
			"groupeJC",
			"importation données brutes de Jonathan Cauchon")
	groupeJC .add_argument(
			"-c",#"--sourcerJC",
			help="importer données brutes",
			action="store_true",
			default=False)
	groupeJC .add_argument(
			"-dc",#"--repertoireJC",
			help="répertoire où elles se trouvent",
			type=str,
			default=DIR_JC)


	groupeJSY = analyseur.add_argument_group(
			"groupeJSY",
			"importation données brutes de Jonathan St-Yves")
	groupeJSY.add_argument(
			"-s",#"--sourcerJSY",
			help="importer données brutes",
			action="store_true",
			default=False)
	groupeJSY .add_argument(
			"-ds",#"--repertoireJSY",
			help="répertoire où elles se trouvent",
			type=str,
			default=DIR_JSY)


	analyseur.add_argument(
			"-X",#"--Xattrib",
			nargs='+',
			type=int,
			choices=XSet,
			help="choix des attributs qui composent les données X",
			default=ATTRIBUTS_X)
	analyseur.add_argument(
			"-y",#"--yattrib",
			type=int,
			nargs='+',
			choices=ySet,
			help="choix des attributs qui composent les étiquettes y",
			default=ATTRIBUTS_y)
	analyseur.add_argument(
			"-k",#"--pkPrefixe",
			type=str,
			help="préfixe de nom à donner au fichier Pickle",
			default="")
	analyseur.add_argument(
			"-p",#"--ptPrefixe",
			type=str,
			help="préfixe de nom à donner aux fichiers PyTorch",
			default="")

	args = analyseur.parse_args()

	# on force l'importation des données et la création d'un fichier
	# binaire Pickle dès que -c ou -s est présent sur la ligne de commande
	importerDonnees = args.c or args.s

	# lancement du flot de traitement selon les
	# options fournies ou celles par défaut
	procederAuTraitement(
		forcerImportation = importerDonnees,
		sourcerJC = args.c,
		repertoireJC = args.dc,
		sourcerJSY = args.s,
		repertoireJSY = args.ds,
		idAttribX = set(args.X), # cast essentiel !
		idAttriby = set(args.y), # cast essentiel !
		pfxPickle = args.k,
		pfxPyTorch = args.p,
	)


if __name__=='__main__':
	r"""
	\brief
	Effectue des tâches seulement si le script est exécuté,
	et non s'il est importé comme module.
	"""
	gererArguments()

else:
	sys.exit(0)


# -----------------------------------------------------------------------------
# EOF