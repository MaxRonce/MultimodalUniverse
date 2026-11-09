# Construire DESI DR1 Main en HATS sur Rusty

Le fork fournit un workflow dédié à **DR1, `SURVEY=main` uniquement**.
SV1, SV2, SV3, commissioning et observations spéciales sont exclus.
Les programmes Main (`dark`, `bright`, `backup`, et `other` si présent) et
les classes galaxies, quasars et étoiles sont conservés.

Les coadds existants sont lus dans :

```text
/mnt/ceph/users/polymathic/external_data/astro/DESI_DR1/
```

Ils ne sont ni copiés, ni modifiés. Aucun transfert Globus n'est nécessaire.
Le vieux `zall-pix-iron.fits` partagé n'a pas de provenance v1 vérifiée.
Chaque campagne récupère donc seulement le **catalogue officiel v1 corrigé**
dans son espace personnel : **22 371 272 640 octets, soit 22,37 Go / 20,84 Gio**
(taille HTTP vérifiée le 9 novembre 2026). Le téléchargement reprend avec curl ;
l'URL, l'ETag, la taille, la date HTTP et le SHA256 sont enregistrés.

Source : [catalogue v1](https://data.desi.lbl.gov/public/dr1/spectro/redux/iron/zcatalog/v1/zall-pix-iron.fits),
[différences v0/v1 et limites DR1](https://data.desi.lbl.gov/doc/releases/dr1/known-issues/).

## 1. Mettre à jour ton clone existant

Depuis le frontend Rusty, pour récupérer cette nouvelle branche la première fois :

```bash
cd "$HOME/MultimodalUniverse"
git status --short
git fetch https://github.com/MaxRonce/MultimodalUniverse.git feat/desi-dr1-main-hats
git switch -c feat/desi-dr1-main-hats FETCH_HEAD
```

Si cette branche existe déjà dans le clone :

```bash
git switch feat/desi-dr1-main-hats
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/desi-dr1-main-hats
```

Conserver les modifications locales si Git refuse le changement de branche.
Le workflow est séparé de LSST/HSC/Legacy et n'utilise pas leurs sorties.
Ne pas changer de branche ni modifier le code pendant une campagne : le build
enregistre son code et ses versions scientifiques et refuse une reprise incompatible.

## 2. Réutiliser l'environnement MMU

```bash
export PATH="$HOME/.local/share/mmu-uv/bin:$HOME/.local/bin:$PATH"
export SETUPTOOLS_SCM_PRETEND_VERSION=0.0.0+hats
uv sync --frozen
.venv/bin/python -c 'import desispec, hats, hats_import, lsdb; print("MMU DESI OK")'
module load slurm
```

Le Python utilisé est celui de `$HOME/MultimodalUniverse/.venv`, pas le petit
environnement de téléchargement Hugging Face. `curl` doit être disponible sur
les nœuds de calcul. Aucun GPU n'est utilisé.

## 3. Lancer d'abord un petit build réel

Il utilise huit coadds Main, produit un vrai HATS et exerce aussi la marge
et la validation. La lecture du catalogue complet reste nécessaire pour
définir proprement la sélection ; cette étape tourne en job Slurm, pas sur le frontend.

```bash
cd "$HOME/MultimodalUniverse"
tmux new -s desi-smoke
bash scripts/desi/launch_dr1_main.sh --smoke --dry-run
bash scripts/desi/launch_dr1_main.sh --smoke
```

Se détacher avec `Ctrl-b`, puis `d`. Le processus Snakemake doit rester vivant
dans tmux pour soumettre la suite des jobs. Les jobs exécutent successivement :

1. téléchargement du petit catalogue v1, sélection et contrôle de présence de
   **tous** les coadds nécessaires ;
2. extraction parallèle et nettoyage des spectres, Parquet intermédiaires atomiques ;
3. assemblage HATS, marge par défaut de **10 arcsec**, validation, publication locale.

Consulter le résultat du test :

```bash
cat /mnt/ceph/users/polymathic/mronceray/desi_dr1_main_smoke/validation.json
```

Il doit indiquer `"status": "PASS"` et `"scope": "smoke"`. Si un job échoue,
ne pas lancer le full avant d'avoir résolu l'erreur du log.

## 4. Lancer le catalogue complet

Le test et le full utilisent des dossiers distincts. Prévoir **au moins 5 To
de quota personnel libre** pour le résultat, les Parquet et le travail HATS.
Le résultat seul est estimé à **1–1,5 To**, à mesurer après le test/full ; le
pic réel dépend du partitionnement et de la compression. `df` affiche le
filesystem partagé, pas ton quota personnel.

```bash
cd "$HOME/MultimodalUniverse"
tmux new -s desi-full
bash scripts/desi/launch_dr1_main.sh --dry-run
bash scripts/desi/launch_dr1_main.sh
```

La partition par défaut est `genx`. Les réservations sont :

| Étape | CPU | Mémoire | Temps limite |
| --- | ---: | ---: | ---: |
| Préparation | 2 | 128 000 Mo | 3 h |
| Extraction, 8 morceaux | 16 par job | 128 000 Mo par job | 12 h par job |
| Assemblage HATS | 8 | 900 000 Mo | 12 h |

Au plus **deux jobs** sont soumis simultanément. Le gros budget mémoire de
l'assemblage reprend le réglage prudent du port MMU DESI existant ; demander
beaucoup de workers Dask avait provoqué des OOM. La disponibilité et les limites
de `genx`/du compte doivent être confirmées par Slurm sur Rusty : le test local
ne vérifie pas l'admission de ces réservations. Exemples d'ajustements explicites :

```bash
export DESI_PARTITION=genx
# Seulement si Slurm demande un compte explicite :
# export DESI_ACCOUNT=nom_du_compte
# Pour limiter davantage la concurrence :
# export DESI_CONCURRENT=1
```

Ne pas lancer `run_production_slurm.sh` pour ce produit : ses chemins sont
ceux de la production partagée et son build DESI historique n'applique pas
les contrôles de cette campagne.

## 5. Suivre et reprendre

```bash
squeue -u "$USER"
tail -f /mnt/ceph/users/polymathic/mronceray/desi_dr1_main/logs/prepare.log
tail -f /mnt/ceph/users/polymathic/mronceray/desi_dr1_main/logs/scatter-000.log
tail -f /mnt/ceph/users/polymathic/mronceray/desi_dr1_main/logs/gather.log
```

Les autres morceaux ont leurs propres `scatter-001.log`, etc. Snakemake affiche
aussi les identifiants et logs Slurm. Pour reprendre **après la fin des jobs
précédents**, relancer exactement la même commande dans tmux. Les groupes
Parquet terminés sont vérifiés puis réutilisés, les écritures interrompues
restent en `.tmp`. Le travail HATS possède un dossier de reprise stable.
Aucun échec de coadd ne peut devenir un succès incomplet silencieux.
Ne pas effacer les sorties, les rapports ou les temporaires pour reprendre.

Pour une nouvelle sélection ou une nouvelle version du code, utiliser une
nouvelle campagne, par exemple `DESI_RUN_ROOT=.../desi_dr1_main_v2`.
Une campagne n'est pas réutilisable avec un nombre de morceaux différent.

## 6. Structure finale et lecture

```text
/mnt/ceph/users/polymathic/mronceray/desi_dr1_main/
├── inputs/                     # catalogue v1 personnel, reçu et SHA256
├── selection.parquet           # sélection exacte ; un row group par coadd
├── plan.json                   # sélection, sources, limites, code et versions
├── preflight.json              # comptages et contrôle des coadds
├── parts/                      # intermédiaires reprenables
├── reports/                    # comptages qualité par morceau
├── logs/
├── hats_work/                  # travail/reprise hats-import
├── dask_tmp/                   # spill Dask sur Ceph, hors /tmp du nœud
├── validation.json             # PASS seulement après les contrôles
└── hats/desi_dr1_main/          # collection HATS publiée après validation
    ├── collection.properties
    ├── README.md
    ├── provenance.json
    ├── desi_dr1_main/           # catalogue principal
    │   ├── hats.properties
    │   ├── partition_info.csv
    │   ├── skymap.fits
    │   └── dataset/            # _metadata, _common_metadata, Norder=.../*.parquet
    └── desi_dr1_main_10arcs/    # marge par défaut
```

La collection est produite par **hats-import**, comme les autres produits
MMU HATS, avec Parquet compressé Zstandard, partitionnement HEALPix,
`_healpix_29`, cartes, métadonnées et marge. Les colonnes spectrales utilisent
un struct PyArrow de listes ordinaires, sans extension Arrow propriétaire.

```bash
cd "$HOME/MultimodalUniverse"
.venv/bin/python - <<'PY'
import lsdb
path = "/mnt/ceph/users/polymathic/mronceray/desi_dr1_main/hats/desi_dr1_main"
catalog = lsdb.open_catalog(path)
print(catalog)
print(catalog.hc_structure.catalog_info.total_rows)
print("Marge chargée :", catalog.margin is not None)
sample = lsdb.open_catalog(
    path, search_filter=lsdb.ConeSearch(ra=150, dec=2, radius_arcsec=3600)
)
# Lire seulement les colonnes nécessaires avant un compute :
print(sample[["ra", "dec", "object_id", "Z", "redshift_good"]].compute().head())
PY
```

Ne pas appeler `catalog.compute()` sur tous les spectres du catalogue complet.
Pour les jointures, utiliser la collection pour charger automatiquement la marge.

## Sélection et qualité du produit

- `SURVEY == main`, `MAIN_PRIMARY`, `OBJTYPE == TGT`, `COADD_FIBERSTATUS == 0`.
- Exclusion du bit DESI `NODATA` (bit 9 de `ZWARN`), des coordonnées non finies
  ou hors domaine et des handles connus manquants du pipeline MMU historique.
- Unicité des `TARGETID` primaires vérifiée ; correspondance exacte avec les
  spectres contrôlée pour chaque fichier. Un coadd manquant requis est bloquant.
- Fusion B/R/Z via `desispec.coadd_cameras` après mise à poids nul des pixels
  invalides. Masque final `True = mauvais`, inverse variance nulle pour ces
  pixels. Flux non finis remplacés par zéro et masqués ; flux mesurés finis
  conservés. Les spectres entièrement masqués sont exclus et comptabilisés.
- **Pas de coupe globale sur le S/N, le type spectral ou le succès du redshift.**
  `ZWARN_BITS` conserve les bits DESI entiers. `ZWARN` reste le booléen du
  loader MMU historique (**True = aucun avertissement**). `redshift_good`
  demande en plus un redshift fini ; ce n'est pas une validation scientifique
  absolue du redshift. Pour un sous-échantillon à redshifts fiables, les coupes
  spécifiques ELG/LRG/QSO/BGS doivent être définies selon l'usage.
- Conservation de `TARGETID`, `SURVEY`, `PROGRAM`, `HEALPIX`, `SPECTYPE`,
  `Z`, `ZERR`, photométrie MMU, S/N B/R/Z et, s'ils existent dans la source,
  `SUBTYPE`, `DELTACHI2`, bits de ciblage, TSNR2 et statistiques de coaddition.
- Spectres de **7 781 pixels**, longueurs d'onde dans le vide en angströms,
  flux dans les unités DESI. Pas de débruitage, normalisation, correction
  d'extinction ni passage au référentiel de repos.
- `spectrum.lsf_sigma` suit l'approximation MMU : sigma gaussien moyen par
  coadd, en pixels, répété sur la grille ; il ne remplace pas la matrice de
  résolution native. Les limites connues de DR1 sur les variances et la
  calibration ne sont pas corrigées par cette conversion.

La validation finale compare tous les comptages aux morceaux, scanne les
colonnes de sélection/flags de chaque partition principale, ouvre la collection
avec LSDB et sa marge, puis contrôle les spectres d'un échantillon de partitions.
Elle n'est pas un audit astrophysique exhaustif de chaque mesure.
Le build complet est à valider sur Rusty ; les tests locaux couvrent de vrais
FITS DESI synthétiques jusqu'à la collection HATS et ses scénarios d'erreur/reprise.

Les données DESI sont sous **CC BY 4.0** : citer DESI DR1, MMU et les
[remerciements DESI](https://data.desi.lbl.gov/doc/acknowledgments/).
