# Steps to Install Docker, PSG, and Required PSG Packages on Ubuntu

1. Install Docker Engine

---

Remove potentially conflicting Docker packages, if they are installed:

.. code-block:: bash

```
sudo apt remove docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc
```

It is normal for this command to report that some or all of these packages are not installed.

Add the official Docker repository:

.. code-block:: bash

```
# Update the package index
sudo apt update

# Install the required dependencies
sudo apt install ca-certificates curl

# Create the directory for repository keys
sudo install -m 0755 -d /etc/apt/keyrings

# Add Docker's official GPG key
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc

sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker's official repository
sudo tee /etc/apt/sources.list.d/docker.sources > /dev/null <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

# Update the package index again
sudo apt update
```

Install Docker Engine and its official plugins:

.. code-block:: bash

```
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Enable and start the Docker service:

.. code-block:: bash

```
sudo systemctl enable --now docker
```

Verify that Docker is running:

.. code-block:: bash

```
sudo systemctl status docker
sudo docker run hello-world
```

Optional: allow the current user to run Docker without `sudo`:

.. code-block:: bash

```
sudo usermod -aG docker "$USER"
```

After running this command, log out and log in again. Alternatively, apply the new group temporarily in the current terminal with:

.. code-block:: bash

```
newgrp docker
```

Verify access without `sudo`:

.. code-block:: bash

```
docker version
```

.. note::

```
Membership in the ``docker`` group provides privileges comparable to root access. Only add trusted users to this group.
```

## 2. Pull the PSG Image

For standard Linux computers with AMD64 or x86-64 processors:

.. code-block:: bash

```
docker logout
docker pull nasapsg/psg-amd
docker tag nasapsg/psg-amd psg
```

The `docker logout` command follows the PSG installation instructions, but it is not normally required unless Docker Hub authentication is causing problems.

Verify that the image was downloaded:

.. code-block:: bash

```
docker image ls psg
```

## 3. Start the PSG Container

Create and start the PSG container:

.. code-block:: bash

```
docker run -d \
    --name psg \
    --restart always \
    -p 127.0.0.1:3000:80 \
    psg
```

The `--restart always` option causes the PSG container to restart automatically after the computer or Docker service is restarted.

Verify that the container is running:

.. code-block:: bash

```
docker ps --filter name=psg
```

The local PSG interface should then be available at:

.. code-block:: text

```
http://localhost:3000
```

A basic HTTP test can also be performed with:

.. code-block:: bash

```
curl -I http://localhost:3000
```

## 4. Update PSG and Install the Required Packages

The following commands are executed on the Ubuntu host. They send package management requests to the PSG service running inside the Docker container.

Update the PSG operational programs for AMD64 systems:

.. code-block:: bash

```
curl "http://localhost:3000/index.php?update=programsamd"
```

Install the correlated-k packages:

.. code-block:: bash

```
curl "http://localhost:3000/index.php?install=corrklowmain"
curl "http://localhost:3000/index.php?install=corrklowtrace"
```

The correlated-k packages can be large and may require considerable download time and disk space.

Package installation and update status can also be checked by opening:

.. code-block:: text

```
http://localhost:3000
```

## 5. Test the Local PSG API

Assuming that a valid PSG configuration file named `psg_cfg.txt` exists in the current directory:

.. code-block:: bash

```
curl --data-urlencode file@psg_cfg.txt http://localhost:3000/api.php
```

For large configuration files:

.. code-block:: bash

```
curl --data-binary @psg_cfg.txt http://localhost:3000/api.php -H "Content-Type: application/octet-stream"
```

## 6. Useful Container Commands

Check the current status:

.. code-block:: bash

```
docker ps -a --filter name=psg
```

View the container processes:

.. code-block:: bash

```
docker top psg
```

View recent logs:

.. code-block:: bash

```
docker logs --tail 100 psg
```

Stop PSG:

.. code-block:: bash

```
docker stop psg
```

Start a stopped PSG container:

.. code-block:: bash

```
docker start psg
```

Restart PSG:

.. code-block:: bash

```
docker restart psg
```

Remove the container:

.. code-block:: bash

```
docker stop psg
docker rm psg
```

## 7. Check Disk Usage

Check the total disk usage managed by Docker:

.. code-block:: bash

```
docker system df
```

Display detailed usage:

.. code-block:: bash

```
docker system df -v
```

Check the writable layer of the PSG container:

.. code-block:: bash

```
docker ps -s --filter name=psg
```

Check the size of the PSG images:

.. code-block:: bash

```
docker image ls nasapsg/psg-amd psg
```

Example from one installation:

* PSG Docker images: approximately **2.406 GB**.
* PSG container with additional packages: approximately **19.53 GB**.

These values are installation-specific. The actual disk usage depends on the PSG image version, installed packages, package updates, temporary files, and
Docker storage configuration.

8. Official Documentation

---

Docker Engine installation instructions:

.. code-block:: text

```
https://docs.docker.com/engine/install/ubuntu/
```

PSG API and local installation instructions:

.. code-block:: text

```
https://psg.gsfc.nasa.gov/helpapi.php#installation
```
