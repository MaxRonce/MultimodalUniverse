# Télécharger les datasets HSC PDR3 sur Rusty

Ce tuto réutilise `uv` et le Python déjà installés pour MMU. Un petit environnement
de téléchargement séparé évite de modifier les dépendances verrouillées de MMU.
Les commandes suivantes sont à exécuter en **Bash sur Rusty**.

## 1. Récupérer les scripts depuis le fork

Si le dépôt MMU existe déjà dans `$HOME/MultimodalUniverse` :

```bash
cd "$HOME/MultimodalUniverse"
git switch feat/lsst-dp2
git pull --ff-only https://github.com/MaxRonce/MultimodalUniverse.git feat/lsst-dp2
```

Si le dépôt n'existe pas encore, le cloner à la place :

```bash
git clone --branch feat/lsst-dp2 --single-branch \
  https://github.com/MaxRonce/MultimodalUniverse.git "$HOME/MultimodalUniverse"
```

Puis charger la configuration depuis le dossier des scripts :

```bash
cd "$HOME/MultimodalUniverse/scripts"
source ./hsc_env.sh
```

Ne pas cloner dans un dossier déjà rempli. Si Git signale des modifications
locales qui bloquent la mise à jour, les conserver avant de poursuivre ; les
commandes ci-dessus n'utilisent ni reset ni mise à jour forcée.

Arborescence créée par ces scripts :

```text
/mnt/ceph/users/polymathic/mronceray/
├── lsst_dp2/                           # données LSST existantes
├── lsst_dp2_raw/
├── hsc_pdr3_raw/
│   ├── .cache/                        # caches uv, Hub et Xet
│   ├── logs/
│   └── tmp/
└── hsc_pdr3/
    ├── hugging-science/mmu_hsc_pdr3_wide_21/
    │   ├── collection.properties
    │   ├── mmu_hsc_pdr3_wide_21/        # catalogue principal HATS
    │   └── mmu_hsc_pdr3_wide_21_10arcs/ # catalogue de marge
    └── UniverseTBD/mmu_hsc_pdr3_dud_22.5/
        ├── collection.properties
        ├── mmu_hsc_pdr3_dud_22.5/
        └── mmu_hsc_pdr3_dud_22.5_10arcs/
```

Ce sont des données **HSC**, distinctes de LSST DP2. L'arborescence des dépôts est
conservée, y compris les fichiers HATS et les marges. `_10arcs` désigne une marge
spatiale de 10 secondes d'arc pour les croisements, pas une autre taille d'image.

Inventaire des fichiers du Hub consulté le 6 novembre 2026 :

| Dépôt complet, marges incluses | Fichiers | Volume décimal |
| --- | ---: | ---: |
| `hugging-science/mmu_hsc_pdr3_wide_21` | 2 867 | 2 809,03 Go |
| `UniverseTBD/mmu_hsc_pdr3_dud_22.5` | 329 | 386,14 Go |
| Total | 3 196 | **3,195 To**, soit environ **2,906 Tio** |

Prévoir environ **3,5 To disponibles dans le quota**, pour les données et une
marge de fonctionnement. `df` et le script affichent l'espace du filesystem ;
cela ne garantit pas que le quota personnel dispose de cet espace.

Sources : [Wide](https://huggingface.co/datasets/hugging-science/mmu_hsc_pdr3_wide_21/tree/main),
[DUD](https://huggingface.co/datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5/tree/main),
[description des marges HATS](https://huggingface.co/datasets/hugging-science/mmu_hsc_pdr3_wide_21/blob/main/README.md).

## 2. Installer uniquement l'outil de téléchargement

```bash
export MMU_REPO="$HOME/MultimodalUniverse"
export MMU_PYTHON="$MMU_REPO/.venv/bin/python"
export MMU_UV="$HOME/.local/share/mmu-uv/bin/uv"

"$MMU_UV" venv --python "$MMU_PYTHON" "$HF_DOWNLOAD_ENV"
"$MMU_UV" pip install --python "$HF_DOWNLOAD_PYTHON" \
  'huggingface_hub>=1.0,<2' 'hf-xet>=1.2,<2'

"$HF_DOWNLOAD_ENV/bin/hf" --version
```

À faire une seule fois ; si cet environnement existe déjà, passer directement à
`uv pip install`. Il n'y a pas besoin de refaire `uv sync` ou de réinstaller MMU.
Le Python MMU continue à servir pour lire et traiter les données ensuite.

## 3. Se connecter à Hugging Face sur Rusty

Créer un token avec permission **Read** dans
[les paramètres Hugging Face](https://huggingface.co/settings/tokens), puis :

```bash
"$HF_DOWNLOAD_ENV/bin/hf" auth login
"$HF_DOWNLOAD_ENV/bin/hf" auth whoami
```

Si un menu propose plusieurs modes de connexion, choisir la connexion par token,
puis coller le token dans l'invite interactive. Répondre **n** si l'outil propose de
l'ajouter aux identifiants Git : le téléchargement passe par l'API HTTP/Xet.
Une connexion dans le navigateur du PC ne connecte pas automatiquement le serveur.
Si `auth whoami` fonctionne déjà sur Rusty, il est inutile de refaire le login.

`hsc_env.sh` ne change pas `HF_HOME` : un token déjà enregistré y reste accessible.
`HF_TOKEN`, si défini dans l'environnement, prend priorité sur le token enregistré.
Le script impose `token=True` et vérifie le compte avant tout téléchargement.
Garder les mêmes variables d'authentification lors d'une reprise.

La connexion permet les requêtes authentifiées et les limites associées au compte.
Le débit dépend aussi du réseau, du service distant et de Ceph ; le login seul
ne garantit pas un multiplicateur de vitesse.

## 4. Voir le volume avant de lancer

```bash
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --dry-run
df -h "$HSC_DEST"
```

Le dry-run récupère uniquement l'inventaire, affiche les commits, les tailles et
les fichiers manquants. Il ne télécharge aucun fichier des dépôts. Il ne remplace
pas la vérification du quota via les outils habituels du cluster.

## 5. Télécharger Wide, puis DUD

Pour un lancement au premier plan :

```bash
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --workers 8
```

Pour continuer après déconnexion SSH, lancer **à la place** :

```bash
LOG="$HSC_RAW/logs/hsc_$(date +%Y%m%d_%H%M%S).log"
nohup "$HF_DOWNLOAD_PYTHON" ./download_hsc.py --workers 8 \
  > "$LOG" 2>&1 < /dev/null &
HSC_PID=$!
echo "$HSC_PID" > "$HSC_RAW/logs/download.pid"
printf 'PID=%s\nLog=%s\n' "$HSC_PID" "$LOG"
tail -f "$LOG"
```

`Ctrl+C` quitte `tail` sans arrêter le téléchargement lancé avec `nohup`.
Après reconnexion, recharger `source ./hsc_env.sh` avant les commandes utilisant
ces variables. Le PID enregistré peut être consulté avec
`ps -p "$(cat "$HSC_RAW/logs/download.pid")" -o pid,etime,args`.
`nohup` protège d'une déconnexion SSH, pas d'un redémarrage ou d'un arrêt par le cluster.
Utiliser un nœud autorisé pour les gros transferts selon les pratiques du cluster.

Le script télécharge le dépôt Wide complet, puis DUD. Les fichiers d'un dépôt sont
téléchargés en parallèle. Pour ne lancer qu'un des deux :

```bash
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --dataset wide --workers 8
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --dataset dud --workers 8
```

Ces deux lignes sont des alternatives au lancement global. Éviter de lancer
plusieurs instances sur les mêmes dossiers.

## 6. Reprendre et vérifier

Après une coupure ou un échec, relancer exactement la même commande :

```bash
source ./hsc_env.sh
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --workers 8
```

Les fichiers déjà terminés sont réutilisés grâce aux métadonnées locales de
Hugging Face. Les transferts interrompus sont repris selon les possibilités du
backend ; un fichier partiel peut devoir être retransféré. Chaque dépôt est figé
sur un commit enregistré dans son `.hf-download-manifest.json` avant les premiers
transferts. Une reprise garde ce commit même si le dépôt distant a évolué.

Conserver `.hf-download-manifest.json` et `.cache/huggingface/` dans chaque dépôt.
Ils permettent respectivement la vérification/reproductibilité et la reprise.
Le script fait jusqu'à cinq tentatives en cas d'erreur, avec une attente progressive.
Il s'arrête directement sur les erreurs HTTP 401, 403 ou 404.

À la fin, il vérifie la présence et la taille de tous les fichiers. Pour refaire
ce contrôle localement, sans authentification ni réseau :

```bash
"$HF_DOWNLOAD_PYTHON" ./download_hsc.py --verify-only
```

Ce contrôle ne recalcule pas les checksums : il détecte les fichiers manquants ou
de taille incorrecte, pas une corruption qui conserverait la même taille.
Si un fichier déjà terminé est signalé avec une taille incorrecte, déplacer ce
fichier hors du dépôt avant de relancer pour forcer sa récupération.

## 7. Réglages de débit

- `hf-xet` est installé et utilisé automatiquement. `HF_XET_HIGH_PERFORMANCE=1`
  est activé par le fichier d'environnement et vise un usage élevé du réseau et
  des ressources de la machine.
- Commencer avec `--workers 8`. Si le réseau et Ceph ne sont pas saturés, essayer
  `--workers 16` au prochain lancement. Plus de workers n'est pas toujours plus rapide.
- Si la machine manque de RAM ou que Ceph peine à suivre, essayer
  `export HF_XET_HIGH_PERFORMANCE=0` puis relancer avec `--workers 4`.
- `local_dir` écrit les données dans leur dossier final, sans créer une seconde
  copie complète dans le cache Hub. Des métadonnées et fichiers temporaires restent
  nécessaires pendant le téléchargement.
- Éviter de commencer par `datasets.load_dataset()` : la préparation Arrow peut
  créer un autre cache volumineux. Ici on récupère directement les fichiers HATS.
- En cas d'erreurs Xet répétées malgré les reprises, tester
  `export HF_HUB_DISABLE_XET=1` avant de relancer, pour utiliser le transport HTTP.

Pas besoin de `git clone`, `git lfs`, `hf_transfer` ou d'extraction d'archives.
Voir [le téléchargement vers un dossier local](https://huggingface.co/docs/huggingface_hub/guides/download)
et [les variables Xet](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables).

## 8. Ouvrir les données avec le Python MMU

Après téléchargement, dans le shell configuré plus haut :

```bash
"$MMU_PYTHON" - <<'PY'
import os
from pathlib import Path
import lsdb

root = Path(os.environ['HSC_DEST'])
wide = lsdb.open_catalog(str(root / 'hugging-science/mmu_hsc_pdr3_wide_21'))
dud = lsdb.open_catalog(str(root / 'UniverseTBD/mmu_hsc_pdr3_dud_22.5'))
print(wide)
print(dud)
PY
```

On ouvre la collection locale pour conserver l'association avec son catalogue de
marge. L'ouverture est paresseuse ; éviter un `.compute()` global sur plusieurs To.
Les [cartes des datasets](https://huggingface.co/datasets/UniverseTBD/mmu_hsc_pdr3_dud_22.5)
décrivent leur structure HATS ; la [documentation LSDB](https://docs.lsdb.io/en/stable/reference/api/lsdb.open_catalog.html)
détaille l'ouverture d'une collection locale.

## Validation des scripts

Vérifiés avec Python 3.12, `huggingface_hub` 1.33.0 et `hf-xet` 1.8.0.
Les cinq tests hors ligne couvrent le dry-run, la reprise sur le commit enregistré,
les tentatives après erreur réseau, le contrôle local et les téléchargements incomplets :

```bash
# Depuis la racine de ce projet, si le dossier tests/ a également été copié.
"$HF_DOWNLOAD_PYTHON" -m unittest discover -s tests -v
```

L'inventaire distant et le téléchargement de quatre petits fichiers de métadonnées
par dépôt ont aussi été vérifiés. Le transfert complet et le débit depuis Rusty
restent à tester sur le serveur.
