#!/usr/bin/env bash
# Descarga los videos de prueba listados en download_links.txt dentro de input_videos/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINKS_FILE="$SCRIPT_DIR/download_links.txt"

if [[ ! -f "$LINKS_FILE" ]]; then
    echo "No se encontró $LINKS_FILE" >&2
    exit 1
fi

n=1
while IFS= read -r url; do
    [[ -z "$url" || "$url" == \#* ]] && continue

    # Nombre de archivo a partir de la URL, sin query string
    fname="$(basename "${url%%\?*}")"
    dest="$SCRIPT_DIR/$fname"

    if [[ -f "$dest" ]]; then
        echo "[$n] Ya existe, se omite: $fname"
    else
        echo "[$n] Descargando: $fname"
        curl -L --fail --silent --show-error -o "$dest" "$url"
    fi
    ((n++))
done < "$LINKS_FILE"

echo "Listo."
