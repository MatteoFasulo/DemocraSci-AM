# ================================================================
# Advanced Dockerfile for RunAI
# Multi-stage build · Non-root user · Pubkey SSH · uv + venv
#
# Usage:
#   docker build -f Dockerfile -t myimage:latest .
#
# Requires: requirements.txt in the build context
# Optional: vscode-extensions.txt in /myhome/ at runtime
# ================================================================

ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:25.06-py3

# ----------------------------------------------------------------
# Stage 1: Dependency Generator (optional but recommended)
# Compiles requirements.txt into a reproducible pinned lockfile.
# To skip this stage, see "Option B" comments in Stage 2 below.
# ----------------------------------------------------------------
FROM python:3.12-slim AS generator
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN uv pip compile pyproject.toml -o requirements.txt
COPY requirements.txt /build/requirements.txt
RUN uv pip compile \
    --python-version 3.12 \
    /build/requirements.txt \
    -o /build/requirements-lock.txt

# ----------------------------------------------------------------
# Stage 2: Runtime Image
# ----------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

ARG PYTHON_VERSION=3.12
ARG APP_USER=appuser
ARG APP_UID=1000
ARG APP_GID=1000
ARG DEBIAN_FRONTEND=noninteractive

# Persist APP_USER as env var so the entrypoint can read it at runtime
ENV APP_USER=${APP_USER}

# ---- System packages ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    openssh-server \
    gosu \
    curl \
    wget \
    git \
    screen \
    vim \
    htop \
    rsync \
    build-essential \
    software-properties-common \
    bubblewrap \
    && rm -rf /var/lib/apt/lists/*

# ---- Python via deadsnakes PPA ----
# NOTE: deadsnakes PPA only works on Ubuntu-based images.
# If using a non-Ubuntu base, install Python another way.
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python${PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

# ---- uv (fast Python package manager) ----
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# ---- Non-root user ----
# Running as root with NFS mounts (/myhome, /mydata) gives the
# container root access to shared filesystems. A non-root user
# limits the blast radius of accidental or malicious writes.
RUN groupadd -g ${APP_GID} ${APP_USER} \
    && useradd -m -u ${APP_UID} -g ${APP_GID} -s /bin/bash ${APP_USER} \
    && echo "${APP_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

# ---- SSH: pubkey-only on port 2222 ----
# Port 2222 avoids conflicts with the host SSH daemon and is the
# standard convention for SSH port-forwarding on the SDSC cluster
# (use `kubectl port-forward` to expose it locally; see docs).
RUN mkdir -p /run/sshd \
    && sed -i 's/#Port 22/Port 2222/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && echo "AllowUsers ${APP_USER}" >> /etc/ssh/sshd_config

# ---- Python venv + dependencies ----
# Installed into the image, NOT on NFS — this makes imports fast.
RUN python${PYTHON_VERSION} -m venv /opt/venv

# Add venv to PATH so it works even without explicit activation
# (e.g., `runai exec` bypasses the entrypoint and .bashrc).
ENV PATH="/opt/venv/bin:${PATH}"

# Option A (default): Install from pinned lockfile (uses Stage 1)
COPY --from=generator /build/requirements-lock.txt /tmp/requirements-lock.txt
RUN pip install --no-cache-dir -r /tmp/requirements-lock.txt \
    && rm /tmp/requirements-lock.txt

# Option B: Install directly without the generator stage.
# Comment out Option A above and uncomment these lines:
#   COPY requirements.txt /tmp/requirements.txt
#   RUN pip install --no-cache-dir -r /tmp/requirements.txt \
#       && rm /tmp/requirements.txt

# ---- Shell config ----
RUN echo 'source /opt/venv/bin/activate' >> /home/${APP_USER}/.bashrc \
    && echo '[ -f /myhome/.bashrc ] && . /myhome/.bashrc' >> /home/${APP_USER}/.bashrc \
    && chown ${APP_USER}:${APP_USER} /home/${APP_USER}/.bashrc

# ---- VS Code server directory ----
RUN mkdir -p /home/${APP_USER}/.vscode-server \
    && chown -R ${APP_USER}:${APP_USER} /home/${APP_USER}/.vscode-server

# ---- Entrypoint ----
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 2222
ENTRYPOINT ["/entrypoint.sh"]