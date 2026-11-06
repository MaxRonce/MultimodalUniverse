# Télécharger Legacy Survey DR10 South sur Rusty

Le script de téléchargement HATS accepte maintenant `--repo-id` et `--root` pour
récupérer un autre dépôt complet. Sans ces options, son comportement HSC reste le même.
L'environnement de téléchargement et le login HF déjà utilisés pour HSC sont réutilisés.

## Volume et durée

L'inventaire **paginé** du dépôt `hugging-science/mmu_legacysurvey_dr10_south_21`,
consulté le 6 novembre 2026 au commit `982f321784d8daf2a5f983eb3ec560f31b90d667`,
contient **148 968 fichiers**, pour **72 757 543 762 232 octets**, soit **72,76 To** :

| Contenu | Volume décimal |
| --- | ---: |
| Catalogue principal, 123 185 970 objets | 67,22 To |
| Catalogue de marge `_10arcs` | 5,54 To |
| Total, fichiers de métadonnées inclus | **72,76 To** |

Prévoir environ **80 To de quota disponible en plus des données HSC/LSST existantes**.
Le chiffre de `df` indique l'espace du filesystem partagé, pas le quota personnel.
Sources : [dépôt et structure HATS](https://huggingface.co/datasets/hugging-science/mmu_legacysurvey_dr10_south_21),
[inventaire API paginé](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree).

Ordres de grandeur si le débit utile vers les fichiers finaux se maintient :

| Débit utile | Durée pour 72,76 To |
| --- | ---: |
| 300 Mo/s | 67 h |
| 500 Mo/s | 40 h |
| 800 Mo/s | 25 h |
| 1 Go/s | 20 h |

Prévoir une allocation de **3 jours**, plutôt que les 12 heures utilisées pour HSC.
Une expiration de l'allocation nécessite une reprise dans une nouvelle allocation.

## 1. Mettre à jour le fork

Sur Rusty :

```bash
cd "$HOME/MultimodalUniverse"
git switch feat/lsst-dp2
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2
```

Le téléchargement Legacy sera placé dans :

```text
/mnt/ceph/users/polymathic/mronceray/
├── legacysurvey_dr10/
│   └── hugging-science/mmu_legacysurvey_dr10_south_21/
│       ├── .hf-download-manifest.json
│       ├── .cache/huggingface/              # métadonnées nécessaires à la reprise
│       ├── collection.properties
│       ├── mmu_legacysurvey_dr10_south_21/
│       └── mmu_legacysurvey_dr10_south_21_10arcs/
└── legacysurvey_dr10_raw/logs/
```

Le cache Xet et les autres caches temporaires sont créés dans le `/tmp` local du
nœud par `legacy_env.sh`. `HF_HOME` reste inchangé pour conserver le login.
Seuls les caches sont placés sur le disque local ; les 72,76 To vont directement sur Ceph.

## 2. Réserver un nœud depuis tmux

Si l'allocation courante dispose encore d'au moins trois jours, on peut la réutiliser.
Sinon, quitter le shell de calcul et l'ancienne allocation après la fin de HSC,
puis lancer sur Rusty dans une session tmux :

```bash
tmux new -s legacy

module load slurm
salloc -p genx -N 1 -n 1 -c 16 \
  --mem=64G --time=3-00:00:00 --constraint=genoa

# Attendre l'attribution de l'allocation, puis :
srun --pty bash -l
hostname
```

Ces ressources sont une configuration de départ issue du téléchargement HSC réussi,
pas une garantie de débit. Les limites et disponibilités du cluster restent applicables.
Il n'y a pas besoin de GPU. Les commandes Slurm sont à lancer manuellement.

## 3. Charger l'environnement sur le nœud de calcul

```bash
cd "$HOME/MultimodalUniverse"
source scripts/legacy_env.sh

# Mettre à jour uniquement l'environnement de téléchargement, pas le Python MMU.
"$HOME/.local/share/mmu-uv/bin/uv" pip install --upgrade \
  --python "$HF_DOWNLOAD_PYTHON" \
  'huggingface_hub>=1.33,<2' 'hf-xet>=1.8,<2'

"$HF_DOWNLOAD_ENV/bin/hf" auth whoami
```

Si l'authentification manque, faire `"$HF_DOWNLOAD_ENV/bin/hf" auth login`.
La version récente du Hub garantit également un inventaire paginé lors du
téléchargement lui-même ; certaines anciennes versions s'appuyaient sur une liste
de fichiers limitée. Le script construit et vérifie aussi son propre inventaire paginé.

## 4. Vérifier le volume puis télécharger

```bash
"$HF_DOWNLOAD_PYTHON" scripts/download_hsc.py \
  --repo-id "$LEGACY_REPO" --root "$LEGACY_DEST" --dry-run
```

L'inventaire devrait compter environ **148 968 fichiers**. Il peut prendre quelques
minutes. On récupère les catalogues principal et de marge, ainsi que les métadonnées.
Après vérification du volume et du quota :

```bash
export HF_XET_HIGH_PERFORMANCE=1
export HF_HUB_DISABLE_PROGRESS_BARS=0
unset HF_HUB_DISABLE_XET TQDM_DISABLE

"$HF_DOWNLOAD_PYTHON" scripts/download_hsc.py \
  --repo-id "$LEGACY_REPO" --root "$LEGACY_DEST" --workers 64
```

Le nom historique `download_hsc.py` est conservé ; `--repo-id` sélectionne uniquement
Legacy. Il ne lance pas les téléchargements HSC dans ce mode.
La sortie doit rester dans le terminal pour les barres Hugging Face ; ne pas la
rediriger avec `nohup`, `>` ou `tee` si l'on souhaite leur affichage interactif.

Pour détacher la session sans arrêter le transfert : **Ctrl+B**, relâcher, puis **D**.
Après reconnexion sur le même login Rusty ayant lancé tmux :

```bash
tmux attach -t legacy
```

Pour arrêter le téléchargement : **Ctrl+C dans le terminal de téléchargement**.
Relancer la même commande pour reprendre. Garder les fichiers terminés,
`.hf-download-manifest.json` et `.cache/huggingface/` du dépôt sur Ceph.
Le script conserve le commit enregistré, même si le dépôt distant évolue.
Un fichier partiel peut devoir être retransféré selon le backend.

À la reprise, la lecture du manifeste puis le contrôle des présences et tailles
des 148 968 fichiers sur Ceph peuvent prendre plusieurs minutes. Le script affiche
désormais ces étapes et une progression environ toutes les cinq secondes lorsque
les appels au filesystem répondent. Les workers ne s'appliquent qu'au téléchargement,
pas à ce contrôle local. Un appel Ceph bloqué peut retarder le prochain message.

Ne pas lancer plusieurs instances sur ce même dépôt. Une répartition sur plusieurs
nœuds nécessite des listes de fichiers disjointes et n'est pas réalisée par ce script.

## 5. Vérification après téléchargement

```bash
"$HF_DOWNLOAD_PYTHON" scripts/download_hsc.py \
  --repo-id "$LEGACY_REPO" --root "$LEGACY_DEST" --verify-only
```

Ce contrôle local vérifie les présences et tailles. Il ne recalcule pas tous les
checksums et ne lit pas les 72,76 To pour valider chaque enregistrement Parquet.

Puis ouvrir le catalogue avec le Python MMU :

```bash
"$HOME/MultimodalUniverse/.venv/bin/python" - <<'PY'
import os
from pathlib import Path
import lsdb

path = Path(os.environ['LEGACY_DEST']) / os.environ['LEGACY_REPO']
catalog = lsdb.open_catalog(str(path), columns=['ra', 'dec', 'object_id'])
print(catalog)
PY
```

L'ouverture est paresseuse. Éviter un `.compute()` global sur l'ensemble du catalogue.

## Validation du nouvel inventaire

L'API `dataset_info(..., files_metadata=True)` renvoyait exactement 100 000 fichiers
pour ce dépôt, ce qui sous-estimait son volume. Le script résout maintenant le commit
puis parcourt intégralement `list_repo_tree(..., recursive=True)` avec sa pagination.
Le mode HSC et les anciens manifestes HSC restent compatibles.

Vérifiés : inventaire complet distant, quatre petits fichiers de métadonnées,
en-tête Parquet d'un fichier principal et d'un fichier de marge, tests hors ligne.
Le téléchargement complet Legacy n'a pas été lancé depuis l'environnement local.
