#!/usr/bin/env bash
# Continue move of genomic.fna / genomic.gtf into raw/{fna,gtf}; drop .fna.gz/.gtf.gz.
set -euo pipefail

ROOT="/home/User14"
SRC="${ROOT}/data/raw/genomes"
DST_FNA="${ROOT}/raw/fna"
DST_GTF="${ROOT}/raw/gtf"

mkdir -p "${DST_FNA}" "${DST_GTF}"

# Finish human: decompress gtf.gz via stdout (hard-linked; gunzip refuses), then remove gz copies
HUMAN_FTP="${SRC}/GCF_000001405.40/ftp"
HUMAN_STEM="GCF_000001405.40_GRCh38.p14_genomic"
if [[ ! -f "${DST_GTF}/${HUMAN_STEM}.gtf" ]]; then
  echo "=== finishing GCF_000001405.40 GTF ==="
  gzip -dc "${HUMAN_FTP}/${HUMAN_STEM}.gtf.gz" > "${DST_GTF}/${HUMAN_STEM}.gtf"
  echo "wrote ${DST_GTF}/${HUMAN_STEM}.gtf"
fi
# Remove .fna.gz and .gtf.gz under human ftp (other hard links elsewhere stay)
for gz in "${HUMAN_FTP}/${HUMAN_STEM}.fna.gz" "${HUMAN_FTP}/${HUMAN_STEM}.gtf.gz"; do
  if [[ -f "${gz}" ]]; then
    rm -v "${gz}"
  fi
done

moved_fna=0
moved_gtf=0
removed_gz=0

for gcf_dir in "${SRC}"/GCF_*; do
  gcf="$(basename "${gcf_dir}")"
  # Skip human — already handled
  if [[ "${gcf}" == "GCF_000001405.40" ]]; then
    continue
  fi
  echo "=== ${gcf} ==="

  fna="$(find "${gcf_dir}" -type f -name '*_genomic.fna' ! -name '*.gz' | head -n 1 || true)"
  if [[ -z "${fna}" ]]; then
    # Already moved?
    if ls "${DST_FNA}/${gcf}"*_genomic.fna >/dev/null 2>&1; then
      echo "FNA already in ${DST_FNA}"
      fna_base="$(basename "$(ls "${DST_FNA}/${gcf}"*_genomic.fna | head -n 1)")"
    else
      echo "ERROR: no uncompressed .fna for ${gcf}" >&2
      exit 1
    fi
  else
    fna_base="$(basename "${fna}")"
    if [[ ! -f "${DST_FNA}/${fna_base}" ]]; then
      mv -v "${fna}" "${DST_FNA}/${fna_base}"
      moved_fna=$((moved_fna + 1))
    else
      echo "skip move FNA (already present): ${fna_base}"
    fi
  fi

  stem="${fna_base%.fna}"
  dest_gtf="${DST_GTF}/${stem}.gtf"

  if [[ ! -f "${dest_gtf}" ]]; then
    gtf="$(find "${gcf_dir}" -type f -name '*genomic.gtf' ! -name '*.gz' | head -n 1 || true)"
    if [[ -n "${gtf}" ]]; then
      mv -v "${gtf}" "${dest_gtf}"
      moved_gtf=$((moved_gtf + 1))
    else
      gtf_gz="$(find "${gcf_dir}" -type f -name '*genomic.gtf.gz' | head -n 1 || true)"
      if [[ -z "${gtf_gz}" ]]; then
        echo "ERROR: no .gtf or .gtf.gz for ${gcf}" >&2
        exit 1
      fi
      echo "Decompressing ${gtf_gz} -> ${dest_gtf}"
      gzip -dc "${gtf_gz}" > "${dest_gtf}"
      moved_gtf=$((moved_gtf + 1))
    fi
  else
    echo "GTF already in ${DST_GTF}"
  fi

  while IFS= read -r gz; do
    rm -v "${gz}"
    removed_gz=$((removed_gz + 1))
  done < <(find "${gcf_dir}" -type f \( -name '*.fna.gz' -o -name '*.gtf.gz' \))
done

echo
echo "Moved FNA (this run): ${moved_fna}"
echo "Moved GTF (this run): ${moved_gtf}"
echo "Removed gz (this run): ${removed_gz}"
echo
echo "=== raw/fna ==="
ls -1 "${DST_FNA}"
echo "=== raw/gtf ==="
ls -1 "${DST_GTF}"
echo
echo "=== genome ID check vs random/genes ==="
mapfile -t genomes < <(ls -1 "${SRC}" | sort)
mapfile -t randoms < <(ls -1 "${ROOT}/random/genes" | sort)
printf 'genomes (%d):\n' "${#genomes[@]}"
printf '%s\n' "${genomes[@]}"
printf 'random/genes (%d):\n' "${#randoms[@]}"
printf '%s\n' "${randoms[@]}"
if [[ "${genomes[*]}" == "${randoms[*]}" ]]; then
  echo "MATCH: genome dirs == random/genes dirs"
else
  echo "MISMATCH"
  exit 2
fi

# Count check
n_fna=$(ls -1 "${DST_FNA}"/*.fna 2>/dev/null | wc -l)
n_gtf=$(ls -1 "${DST_GTF}"/*.gtf 2>/dev/null | wc -l)
echo "FNA count: ${n_fna}; GTF count: ${n_gtf}"
remaining_gz=$(find "${SRC}" -type f \( -name '*.fna.gz' -o -name '*.gtf.gz' \) | wc -l)
echo "Remaining .fna.gz/.gtf.gz under genomes: ${remaining_gz}"
