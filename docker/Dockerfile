FROM nvcr.io/nvidia/pytorch:25.01-py3

# System packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y net-tools tmux sudo && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# GitHub CLI (gh)
RUN mkdir -p -m 755 /etc/apt/keyrings && \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        -o /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list && \
    apt-get update && apt-get install -y gh && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python packages
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Create dev user matching host UID/GID
ARG HOST_UID=1000
ARG HOST_GID=1000
RUN EXISTING_USER=$(getent passwd ${HOST_UID} | cut -d: -f1) && \
    if [ -n "$EXISTING_USER" ]; then \
        usermod -l devuser -d /home/devuser -m "$EXISTING_USER"; \
    else \
        EXISTING_GROUP=$(getent group ${HOST_GID} | cut -d: -f1) && \
        [ -z "$EXISTING_GROUP" ] && groupadd -g ${HOST_GID} devgroup || true && \
        useradd -m -u ${HOST_UID} -g ${HOST_GID} -s /bin/bash devuser; \
    fi && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/devuser && \
    chmod 0440 /etc/sudoers.d/devuser

# Entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace
USER devuser
