#!/usr/bin/env bash
# Build Demon of Ignorance from the vendored source. javac does not copy
# resources, and on modern JDKs DoI's classpath-resource fallback
# (Class.class.getResourceAsStream) returns null across module boundaries,
# so ai.cfg must also sit in the engine's working directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/vendor/stratego"
mkdir -p bin
find src -name '*.java' >/tmp/doi-sources.txt
javac -d bin -cp src @/tmp/doi-sources.txt 2>/dev/null || javac -d bin -cp src @/tmp/doi-sources.txt
rsync -a --exclude='*.java' src/com/ bin/com/
cp src/com/cjmalloy/stratego/resource/ai.cfg ai.cfg
echo "DoI built: $(ls bin/com/cjmalloy/stratego/player/*.class | wc -l | tr -d ' ') player classes"
