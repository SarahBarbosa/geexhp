Steps to Install Docker, PSG, and Specific Packages on Ubuntu
=============================================================

1. Install Docker
------------------

.. code-block:: bash

    # Update the system
    sudo apt update && sudo apt upgrade
    
    # Install the dependencies
    sudo apt install apt-transport-https ca-certificates curl software-properties-common
    
    # Add GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
    
    # Add the Docker repository
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    
    # Install Docker
    sudo apt install docker-ce docker-ce-cli containerd.io

    # Start Docker
    sudo systemctl start docker
    sudo usermod -aG docker ${USER}

When adding the GPS key, you might need to use the command below instead:

.. code-block:: bash

    # Add GPG key
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/docker-archive-keyring.gpg

2. Pull the PSG Image
----------------------
You might need run these commands as sudo.

.. code-block:: bash

    docker logout
    docker pull nasapsg/psg-amd
    docker tag nasapsg/psg-amd psg

3. Start the PSG Container
---------------------------

.. code-block:: bash

    docker run -d --name psg -p 127.0.0.1:3000:80 psg

4. Update the PROGRAMSAMD Package and install Specific Packages Inside the Container
--------------------------------------------------

.. code-block:: bash

    curl http://localhost:3000/index.php?update=programsamd
    curl http://localhost:3000/index.php?install=corrklowmain
    curl http://localhost:3000/index.php?install=corrklowtrace

For more information, see the official documentation: https://psg.gsfc.nasa.gov/helpapi.php#installation

Disk Usage:
-----------

- The images take up **2.406 GB**.
- The active container is using **19.53 GB**, which includes PSG and installed packages.
