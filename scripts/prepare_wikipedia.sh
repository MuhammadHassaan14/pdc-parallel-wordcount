#!/usr/bin/env bash
# prepare_wikipedia.sh - download an English Wikipedia dump and turn it into
# the plain-text corpus used by the programs (one article per line).
#
# Usage:
#   bash scripts/prepare_wikipedia.sh            # full enwiki dump (~22 GB download, many hours)
#   bash scripts/prepare_wikipedia.sh <dump-url> # any other dump, e.g. one of the
#                                                # enwiki-latest-pages-articlesN.xml-p...bz2 parts
#
# Partial dump files are listed at https://dumps.wikimedia.org/enwiki/latest/
# Two or three of the "pages-articlesN" parts are enough for a 1-2 GB corpus.
#
# Requirements: wget, python3, pip install wikiextractor
# NOTE: wikiextractor 3.0.6 crashes on Python 3.11+ ("global flags not at the
# start of the expression"). Use a Python 3.10 virtualenv for this step.
set -e
URL=${1:-https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2}
NPROC=$(nproc)
mkdir -p data/raw
DUMP=data/raw/$(basename "$URL")

if [ ! -f "$DUMP" ]; then
  echo ">> downloading $URL"
  wget -c -O "$DUMP" "$URL"
fi

echo ">> extracting plain text with WikiExtractor ($NPROC processes)"
python3 -m wikiextractor.WikiExtractor "$DUMP" --json -b 100M --processes "$NPROC" -o data/raw/extracted

echo ">> converting to one-article-per-line format"
python3 scripts/wiki_to_lines.py data/raw/extracted data/enwiki.txt

echo ">> creating subsets"
python3 scripts/make_subsets.py data/enwiki.txt 256M 1G 2G 5G
ls -lh data/*.txt
