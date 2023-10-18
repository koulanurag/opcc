=============
Installation
=============

````````````
Setup Mujoco
````````````
+ Download `[mujoco 210] <https://github.com/google-deepmind/mujoco/releases/tag/2.1.0>`_ and unzip in  `~/.mujoco`
+ Add following to `.bashrc/.zshrc` and source it.

  .. code-block:: console

      $ export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210/
      $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

  (You can also refer `here <https://github.com/koulanurag/opcc/blob/main/.github/workflows/python-package.yml#L41>`_ for step-by-step instructions on mujoco installation)

`````````````````
Install package
`````````````````
#. Setup `[Python 3.7+] <https://www.python.org/downloads/>`_ and optionally(recommended) create a  `virtualenv` (`refer here <https://docs.python.org/3/tutorial/venv.html>`_)
#. Install following

    .. code-block:: console

       $ pip3 install 'setuptools<=66'
       $ pip3 install 'wheel<=0.38.4'

#. Install `Pytorch>=1.8.0 <https://pytorch.org/>`_.
#. Install :code:`opcc`:

   .. code-block:: console

       $ pip install git+https://github.com/koulanurag/opcc@main#egg=opcc

