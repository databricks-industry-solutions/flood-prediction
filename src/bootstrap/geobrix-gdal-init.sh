#!/bin/bash
# Flood demo - self-contained GeoBrix v0.2.0 init script
#
# Installs GDAL native libraries, downloads the GeoBrix JAR + GDAL JNI shared
# object from the public GitHub release, and stages them for the Spark driver
# and executors. Does NOT require a Databricks Volume (it fetches artifacts
# at cluster startup), which keeps this demo deployable on workspaces whose
# default storage binding can't create managed volumes.
#
# NOTE: We deliberately do NOT use `set -e` here. The upstream
# databrickslabs/geobrix script tolerates transient nonzero exits from
# add-apt-repository (e.g. launchpad warnings) and we mirror that.
set -uxo pipefail

GEOBRIX_VERSION="0.2.0"
RELEASE="https://github.com/databrickslabs/geobrix/releases/download/v${GEOBRIX_VERSION}"

# ---- software-properties-common FIRST (provides add-apt-repository) ----
# Must be installed before any add-apt-repository call.
sudo apt-get update -y || true
sudo apt-get install -y software-properties-common

# ---- ubuntugis PPA only ----
# DBR 17.3 runs Ubuntu 24.04 (Noble), which already configures
# main/universe/multiverse/restricted for noble + noble-updates + noble-security
# + noble-backports via /etc/apt/sources.list.d/ubuntu.sources (deb822 format).
# The upstream geobrix-gdal-init.sh re-adds those sources via the legacy `deb`
# format; on Noble that creates duplicate-source warnings and an apt-get update
# retry loop that can hang the init script for 30+ minutes. We skip them.
sudo add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
sudo apt-get update -y

# ---- GDAL natives ----
sudo apt-get -o DPkg::Lock::Timeout=-1 install -y \
    unixodbc libcurl3-gnutls libsnappy-dev libopenjp2-7
sudo apt-get -o DPkg::Lock::Timeout=-1 install -y \
    libgdal-dev gdal-bin python3-gdal

# ---- Python bindings for GDAL ----
pip install --upgrade pip setuptools wheel cython
pip install wheel "setuptools==74.0.0" "numpy==2.1.3"
export GDAL_CONFIG=/usr/bin/gdal-config
pip install --no-cache-dir --force-reinstall "GDAL[numpy]==$(gdal-config --version).*"

# ---- GeoBrix JAR + GDAL JNI shared object (from GitHub releases) ----
# Fail loudly if the downloads don't succeed - these are required.
TMP=$(mktemp -d)
cd "$TMP"
curl -fSL --retry 5 --retry-delay 5 -o geobrix.jar       "${RELEASE}/geobrix-${GEOBRIX_VERSION}-jar-with-dependencies.jar" \
  || { echo "FATAL: failed to download geobrix jar"; exit 1; }
curl -fSL --retry 5 --retry-delay 5 -o libgdalalljni.so "${RELEASE}/libgdalalljni.so" \
  || { echo "FATAL: failed to download libgdalalljni.so"; exit 1; }

sudo cp libgdalalljni.so /usr/lib/libgdalalljni.so
sudo mkdir -p /databricks/jars
sudo cp geobrix.jar      /databricks/jars/geobrix-${GEOBRIX_VERSION}.jar

cd / && rm -rf "$TMP"
echo "GeoBrix ${GEOBRIX_VERSION} init script finished."
