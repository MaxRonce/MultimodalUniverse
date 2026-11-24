# Publier JADES JWST depuis Rusty

Ce guide part du **HATS déjà terminé**, sans relancer l'extraction ni la conversion.
Il prépare une collection documentée, vérifie les 484 219 sources exactes, puis
effectue un upload Hugging Face reprenable dans un dépôt privé. La publication
publique et l'intégration au namespace MMU restent des opérations distinctes.

## 1. Récupérer les outils

Depuis le checkout `MultimodalUniverse-jades`, récupérer le commit de publication
indiqué dans le message qui accompagne ce guide :

```bash
git status --short
git fetch origin codex/jades-hats-recovery
# Remplacer COMMIT_PUBLICATION par le SHA fourni ; une seule fois.
git cherry-pick COMMIT_PUBLICATION
```

Le répertoire doit être propre avant le cherry-pick. S'il est déjà appliqué, ne
pas le répéter. Cette méthode convient au checkout Rusty qui possède ses propres
commits ; un `git pull --ff-only` peut échouer si les historiques ont divergé.
Le code du pipeline ayant produit les données est
[figé ici](https://github.com/MaxRonce/MultimodalUniverse/tree/6c6b7aba1d3ec5e741af69fb58da8460dbb247d8).
Les outils de ce guide ajoutent le conditionnement et la validation de publication.

## 2. Choisir la collection réellement présente

```bash
export JADES_RUN=/mnt/ceph/users/polymathic/mronceray/jades_dr5_runs
export JADES_COLLECTION=/mnt/ceph/users/polymathic/MultimodalUniverse_v1.6/jades-jwst

if [[ ! -f "$JADES_COLLECTION/collection.properties" ]]; then
  export JADES_COLLECTION="$JADES_RUN/hats_jades_jwst_20261113T181400Z/jades_jwst"
fi
test -f "$JADES_COLLECTION/collection.properties" || { echo 'Collection introuvable'; false; }
test -f "$JADES_COLLECTION/jades_jwst/hats.properties" || { echo 'Catalogue principal absent'; false; }
printf 'Collection : %s\n' "$JADES_COLLECTION"

export JADES_MANIFEST="$JADES_RUN/jades_dr5_manifest.json"
export JADES_COORDINATES="$JADES_RUN/scratch_jades_jwst_full_dist_v2.preflight-coordinates.npz"
test -f "$JADES_MANIFEST" && test -f "$JADES_COORDINATES"
export JADES_RELEASE_STAGE="$JADES_RUN/releases/jades-jwst-dr5-v1"
mkdir -p "$JADES_RUN/logs"
```

**Arrêter si un test échoue.** La racine de collection contient
`collection.properties`, pas `hats.properties`. Ne pas déplacer seulement le
catalogue principal : la marge et les références relatives doivent l'accompagner.
Le premier déplacement n'avait pas été confirmé ; ces commandes détectent les
deux emplacements possibles sans déplacer ni écraser les données.

## 3. Valider et préparer le dossier de publication

Lancer sur un nœud de calcul. Le script lit les pieds de page et quatre colonnes
scalaires des 527 partitions, jamais les cubes image. Il exige :

- 484 219 lignes, identifiants uniques et hash exact de sélection ;
- 303 762 GOODS-S et 180 457 GOODS-N ;
- égalité exacte des IDs, champs et coordonnées avec le cache préflight ;
- schémas identiques et lecture HATS de la collection source puis déplacée.

```bash
PREP_JOB=$(sbatch --parsable --export=ALL --partition=genx \
  --job-name=jades-release-check --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=16G --time=04:00:00 \
  --output="$JADES_RUN/logs/release_prepare_%j.out" \
  --wrap='set -euo pipefail; unset VIRTUAL_ENV; uv run python -m scripts.jades.prepare_release --collection "$JADES_COLLECTION" --stage "$JADES_RELEASE_STAGE" --manifest "$JADES_MANIFEST" --coordinates "$JADES_COORDINATES"')
echo "$PREP_JOB"
```

Après la fin du job :

```bash
sacct -j "$PREP_JOB" --format=JobID,State,Elapsed,ExitCode
tail -n 45 "$JADES_RUN/logs/release_prepare_${PREP_JOB}.out"
cat "$JADES_RELEASE_STAGE/release/provenance.json"
```

Continuer seulement si le job est `COMPLETED`, code `0:0`, et affiche `READY`.
Le stage est créé exclusivement : une préparation interrompue peut laisser un
dossier partiel, dans ce cas choisir un **nouveau** chemin `--stage` pour retenter.
Ne pas confondre ceci avec une reprise d'upload : celle-ci réutilise le stage terminé.

Les Parquet sont des **liens physiques**, donc pas une seconde copie des images.
Cela nécessite le même système de fichiers et l'autorisation de créer ces liens.
Conserver source et stage immuables : une modification d'un Parquet affecterait
les deux chemins. Les métadonnées sont copiées séparément et peuvent être éditées
dans le stage. Le stage n'est pas une sauvegarde indépendante.

Le dossier prêt contient :

```text
jades-jwst-dr5-v1/
  README.md                      # dataset card anglaise
  collection.properties
  jades_jwst/                    # catalogue principal
  <répertoire de marge>/         # nom original conservé
  release/
    input_manifest.json
    provenance.json             # validation, taille réelle, liens et hash du code
    collection_files.json       # chemins et tailles de toute la collection
    prepare_release.py
    PUBLISHING.md
```

La taille finale est `collection_bytes`, incluant la marge. Les 961 Gio évoqués
pour le scratch ne sont pas une mesure de la taille finale. Vérifier le quota
et les limites de stockage du compte/organisation HF pour **cette taille** avant
le transfert : [documentation des limites](https://huggingface.co/docs/hub/storage-limits).
Ne pas téléverser le scratch, les caches préflight, les logs Slurm ou les credentials.
Inspecter le manifest et la card pour les informations locales que l'on souhaite exposer.

## 4. Licence DR5 et dataset card

La card est [DATASET_CARD.md](DATASET_CARD.md), copiée en `README.md` à la racine
du stage. Elle documente sélection, masques, unités, provenance, biais et usage
AION-2 envisagé. Lire le fichier généré avant upload.

[MAST](https://archive.stsci.edu/hlsp/jades#citations) indique CC BY 4.0 pour ses
produits JADES HLSP. Pour les fichiers DR5 précis venant du
[portail UCSC](https://slate.ucsc.edu/~brant/jades-dr5/) et de la
[release MIRI](https://github.com/staceyalberts/JWST-JADES-MIRI-PARALLELS/releases/tag/v1.0),
conserver une preuve explicite que ces conditions s'appliquent aussi à ces produits.
La préparation ne prétend pas résoudre cette question et ne renseigne pas
automatiquement `license: cc-by-4.0` dans les métadonnées HF.

Avant mise en public :

1. Obtenir la déclaration DR5 ou confirmation des responsables, archiver la preuve
   dans `release/LICENSE_EVIDENCE.md` avec URL, date, produits couverts et conditions.
2. Si CC BY 4.0 est confirmé pour tous les produits inclus, ajouter
   `license: cc-by-4.0` au YAML du README, fournir les termes applicables et retirer
   les mentions de licence/publication en attente. Si les termes diffèrent,
   documenter ceux réellement applicables ; ne pas attribuer une licence supposée.
3. Compléter la bibliographie et les acknowledgements demandés par le portail DR5.
   La card inclut déjà Johnson, Robertson et Alberts (2026) et les liens de code.
4. Mettre à jour `license_status` dans `release/provenance.json` **avant le premier
   upload**, puis garder cette provenance stable pendant les reprises.

L'upload privé peut être effectué pendant cette vérification. Le passage sous
`MultimodalUniverse/` suppose les droits d'écriture et l'accord des mainteneurs MMU.
Une licence MIT du code ne donne pas la licence des données.

## 5. Authentification et dépendance d'upload

Depuis le checkout, sur le login node avec accès Internet :

```bash
# Environnement distinct : ne modifie pas le lock du pipeline scientifique.
uv venv "$JADES_RUN/hf-upload-env" --python 3.11
uv pip install --python "$JADES_RUN/hf-upload-env/bin/python" 'huggingface_hub==0.36.0'
export HF_HOME="$HOME/.cache/huggingface"
"$JADES_RUN/hf-upload-env/bin/hf" auth login
"$JADES_RUN/hf-upload-env/bin/hf" auth whoami

# Choisir UN dépôt, puis conserver exactement cet identifiant pour les reprises.
export HF_REPO_ID=MaxRonce/jades-jwst
# Si les mainteneurs ont autorisé le namespace et accordé les droits :
# export HF_REPO_ID=MultimodalUniverse/jades-jwst
```

Utiliser un token HF autorisé à écrire dans le dépôt choisi. Le saisir au prompt,
jamais dans une commande `sbatch`, le dépôt Git ou un log. Le compte privé doit
disposer du stockage nécessaire. Ne pas recréer un environnement déjà prêt.

## 6. Uploader via Slurm, sans dépendre du terminal

```bash
UPLOAD_JOB=$(sbatch --parsable --export=ALL --partition=genx \
  --job-name=jades-hf-upload --nodes=1 --ntasks=1 \
  --cpus-per-task=4 --mem=16G --time=24:00:00 \
  --output="$JADES_RUN/logs/hf_upload_%j.out" \
  --wrap='set -euo pipefail; export HF_XET_CACHE="${SLURM_TMPDIR:-/tmp}/jades-xet-${SLURM_JOB_ID}"; "$JADES_RUN/hf-upload-env/bin/python" -m scripts.jades.publish_release upload --stage "$JADES_RELEASE_STAGE" --repo "$HF_REPO_ID"')
printf 'Upload job : %s\n' "$UPLOAD_JOB"
printf 'STAGE=%s\nREPO=%s\nUPLOAD_JOB=%s\n' "$JADES_RELEASE_STAGE" "$HF_REPO_ID" "$UPLOAD_JOB" \
  > "$JADES_RUN/logs/hf_upload_launch.txt"
```

Le nœud choisi doit avoir accès à Hugging Face et au cache d'authentification.
Si l'accès sortant est interdit, demander le nœud de transfert adapté à Rusty.
L'upload lit toutes les images et peut durer longtemps. Le cache Xet utilise le
disque local ; surveiller sa capacité. Le dossier `.cache/huggingface` du stage
conserve l'avancement de `upload_large_folder`, ne pas le supprimer en cours de reprise.
Voir [le guide HF](https://huggingface.co/docs/huggingface_hub/guides/upload).

Le script crée le dépôt **privé**, refuse d'envoyer vers un dépôt déjà public ou
un dépôt contenant une autre release, et ne change jamais la visibilité.
En cas de timeout/échec réseau, relancer le même bloc avec **le même stage et le
même repo**. Il reprend les tâches déjà enregistrées. Un échec persistant de
quota/authentification exige d'abord de corriger cette cause.

```bash
sacct -j "$UPLOAD_JOB" --format=JobID,State,Elapsed,ExitCode
tail -n 60 "$JADES_RUN/logs/hf_upload_${UPLOAD_JOB}.out"
```

Après reconnexion, restaurer les variables de l'étape 2 et `HF_REPO_ID`, ou lire
`logs/hf_upload_launch.txt`. Le crash du terminal n'arrête pas le job Slurm.

## 7. Contrôler le dépôt puis publier

```bash
"$JADES_RUN/hf-upload-env/bin/python" -m scripts.jades.publish_release verify \
  --stage "$JADES_RELEASE_STAGE" --repo "$HF_REPO_ID"
```

Ce contrôle compare tous les chemins/tailles du stage et la provenance distante,
et imprime le **SHA de révision HF**. Ce n'est pas une relecture intégrale de
tous les pixels ni une vérification indépendante des checksums de tout le payload.
Le Hub contrôle les objets transférés ; pour une vérification supplémentaire,
télécharger des partitions témoins dans un autre dossier et comparer leur SHA256
au fichier local correspondant, puis les ouvrir avec PyArrow.

Consulter le README rendu sur HF, les références de collection, les rapports et
les citations. Après confirmation de la licence et relecture MMU, effectuer les
dernières corrections documentaires par `hf upload` ciblé si le dépôt privé existe
déjà (le preparatory uploader refuse une provenance différente), puis vérifier de
nouveau. Passer le dépôt en public depuis ses **Settings** seulement lorsque la
card décrit les conditions confirmées et le bon statut de release.

Si une publication initiale a été faite sous le compte personnel, utiliser la
fonction de transfert HF vers l'organisation avec les mainteneurs ; ne pas créer
une seconde copie incohérente des mêmes données. Après transfert, enregistrer
l'identifiant définitif et le SHA final HF dans la configuration AION-2.

Pour l'entraînement : lire `jades_jwst/` avec LSDB/MMU, conserver la marge pour les
jointures, définir les splits par source/région, contrôler le recouvrement avec
les autres surveys, puis valider un batch complet (bandes, masques, échelles,
labels manquants). Ne pas concaténer récursivement les Parquet de la collection :
cela compterait aussi les copies de marge.
