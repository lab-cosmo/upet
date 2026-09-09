.. _usage_torchsim:

TorchSim
========

.. note::

   Full documentation for the UPET integration with TorchSim is work in progress.

UPET models can be used as a force field inside `TorchSim
<https://github.com/Radical-AI/torch-sim>`_ via the
``metatomic`` engine. See the `metatomic TorchSim documentation
<https://docs.metatensor.org/metatomic/latest/engines/torchsim.html>`_ for the
general workflow; exporting a UPET checkpoint with ``mtt export`` (see
:ref:`usage_metatrain`) produces a TorchScript model that can be plugged
directly into the TorchSim driver.
